import argparse
import logging
import os
from typing import Set

import numpy as np
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils.data_utils import code_prompt_completion_converter
from utils.helper import naming_conversion
from utils.general_utils import configure_trainable_layers


global_rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


master_process = global_rank == 0
distributed = world_size > 1


def load_filtered_ids(filtered_id_path: str) -> Set[int]:
    with np.load(filtered_id_path, allow_pickle=False) as data:
        filtered_ids_set = set(data["filtered_ids"].tolist())
    return filtered_ids_set


def main(args):
    # Build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training dataset
    raw_data = load_dataset(f"ise-uiuc/{args.task}", split="train")

    if args.filter_length:
        tmp_id_path = os.path.join('auxiliary_data', f'{args.task}_{args.base_model}_filtered_ids.npz')
        filtered_train_ids = load_filtered_ids(tmp_id_path)
        raw_data = raw_data.filter(lambda _, idx: idx not in filtered_train_ids, with_indices=True)

    formatted_trainset = raw_data.map(code_prompt_completion_converter,
                                      batched=True,
                                      remove_columns=list(raw_data.column_names))
    shuffled_formatted_trainset = formatted_trainset.shuffle(seed=args.seed + 11)

    # training args
    sft_config = SFTConfig(output_dir=args.save_dir,
                           model_init_kwargs={'dtype': 'float32'},
                           per_device_train_batch_size=args.batch_size,
                           per_device_eval_batch_size=args.eval_batch_size,
                           gradient_accumulation_steps=args.grad_accum,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           adam_beta1=args.adam_beta1,
                           adam_beta2=args.adam_beta2,
                           max_grad_norm=args.max_grad_norm,
                           num_train_epochs=args.epochs,
                           warmup_steps=args.warmup_ratio,
                           optim="adamw_torch_fused",
                           lr_scheduler_type=args.lr_scheduler_type,
                           lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                           save_strategy='epoch' if args.save_checkpoint else 'no',
                           bf16=True,
                           tf32=True,
                           deepspeed=args.deepspeed_config_file,
                           gradient_checkpointing=args.enable_gradient_checkpointing,
                           gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=1,
                           dataloader_prefetch_factor=2,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='no',
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=16,
                           max_length=args.max_seq_length,
                           pad_to_multiple_of=8,
                           completion_only_loss=True,)

    sft_trainer = SFTTrainer(model=args.model_name_or_path,
                             args=sft_config,
                             train_dataset=shuffled_formatted_trainset,
                             processing_class=tokenizer, )

    configure_trainable_layers(model=sft_trainer.model,
                               last_n=args.last_n)

    sft_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Mistral-7B-v0.1', 'deepseek-coder-6.7b-base'],
                        default='deepseek-coder-6.7b-base')
    parser.add_argument('--task', type=str,
                        choices=['Magicoder-OSS-Instruct-75K'],
                        default='Magicoder-OSS-Instruct-75K')

    # full finetuning relevant
    parser.add_argument('--last-n', type=int, required=True)

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--logging-steps', type=int, default=10)

    parser.add_argument('--deepspeed-config-file', 
                        type=str,
                        default='configs/ds_configs/stage3_no_offloading_accelerate.conf')

    # training set size
    parser.add_argument('--filter-length', action='store_true')
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # logging
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--save-checkpoint', action='store_true')

    args = parser.parse_args()

    if 'Mistral' in args.base_model:
        args.model_name_or_path = f'mistralai/{args.base_model}'
    elif 'deepseek' in args.base_model:
        args.model_name_or_path = f'deepseek-ai/{args.base_model}'
    else:
        raise ValueError

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = ''

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    else:
        args.lr_scheduler_kwargs = {}

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}

    save_dir = os.path.join(args.save_root, f"{args.base_model}-fp32", args.task, f'FT-L{args.last_n}{args.suffix}',
                            f'LR-{naming_conversion(args.learning_rate_exponent)}')

    os.makedirs(save_dir, exist_ok=True)

    if master_process:
        logging.info(args)

    args.save_dir = save_dir
    set_seed(args.seed)

    main(args)
