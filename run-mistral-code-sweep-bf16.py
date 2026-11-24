import argparse
import logging
import os
import random
from typing import List

import torch
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from utils.data_utils import code_prompt_completion_converter
from utils.file_utils import load_filtered_ids
from utils.helper import naming_conversion


ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    ddp_rank = int(os.environ['RANK'])
    master_process = ddp_rank == 0
else:
    master_process = True


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

    # Create Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 dtype=torch.bfloat16,)

    selected_layers: List[int] = None
    model_n_layer = model.config.num_hidden_layers
    if args.last_n is not None:
        if args.last_n == 0:
            model.requires_grad_(True)
        else:
            assert 0 < args.last_n <= model_n_layer, "Last N must be between 1 and {}".format(model_n_layer)
            selected_layers = list(range(model_n_layer - args.last_n, model_n_layer))
    elif args.first_n is not None:
        assert 0 < args.first_n <= model_n_layer, "First N must be between 1 and {}".format(model_n_layer)
        selected_layers = list(range(args.first_n))
    else:
        num_ft_layers = args.sample_n_layers
        assert 0 < num_ft_layers <= model_n_layer, "Number of finetuned layers must be between 1 and {}".format(model_n_layer)

        tmp_rng = random.Random(args.sample_n_seed)
        selected_layers = tmp_rng.sample(range(model_n_layer), num_ft_layers)

    if selected_layers is not None:
        if args.ft_mode == 'attn':
            prefix_list = [f"model.layers.{l_idx}.self_attn." for l_idx in selected_layers]
        elif args.ft_mode == 'mlp':
            prefix_list = [f"model.layers.{l_idx}.mlp." for l_idx in selected_layers]
        else:
            prefix_list = [f"model.layers.{l_idx}." for l_idx in selected_layers]

        model.requires_grad_(False)
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefix_list):
                p.requires_grad_(True)

    # training args
    sft_config = SFTConfig(output_dir=args.save_dir,
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
                           lr_scheduler_type=args.lr_scheduler_type,
                           lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                           save_strategy='epoch' if args.save_checkpoint else 'no',
                           bf16=True,
                           tf32=True,
                           gradient_checkpointing=args.enable_gradient_checkpointing,
                           gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=8,
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

    sft_trainer = SFTTrainer(model=model,
                             args=sft_config,
                             train_dataset=shuffled_formatted_trainset,
                             processing_class=tokenizer, )

    sft_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Mistral-7B-v0.1', 'deepseek-coder-6.7b-base'],
                        default='Mistral-7B-v0.1')
    parser.add_argument('--task', type=str,
                        choices=['Magicoder-OSS-Instruct-75K'],
                        default='Magicoder-OSS-Instruct-75K')

    # full finetuning relevant
    ft_group = parser.add_mutually_exclusive_group(required=True)
    ft_group.add_argument('--last-n', type=int)
    ft_group.add_argument('--first-n', type=int)
    ft_group.add_argument('--sample-n', type=str)   # format: number of layers-S{seed}

    parser.add_argument('--ft-mode', type=str,
                        choices=['all', 'attn', 'mlp'],
                        default='all')

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--logging-steps', type=int, default=10)

    # training set size
    parser.add_argument('--filter-length', action='store_true')
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # logging
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--save-checkpoint', action='store_true')

    args = parser.parse_args()

    if args.sample_n is not None:
        sample_n_parts = args.sample_n.split('-S')
        if len(sample_n_parts) != 2 or not sample_n_parts[0].isdigit() or not sample_n_parts[1].isdigit():
            parser.error("--sample-n must use format {num_layers}-S{seed}, e.g. 4-S42")
        args.sample_n_layers = int(sample_n_parts[0])
        args.sample_n_seed = int(sample_n_parts[1])

    if 'Mistral' in args.base_model:
        args.model_name_or_path = f'mistralai/{args.base_model}'
    elif 'deepseek' in args.base_model:
        args.model_name_or_path = f'deepseek-ai/{args.base_model}'
    else:
        raise ValueError

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = '' if args.ft_mode == 'all' else f"-{args.ft_mode}"

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    else:
        args.lr_scheduler_kwargs = {}

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}

    if args.last_n is not None:
        assert args.last_n > 0 or args.ft_mode == 'all'
        save_dir = os.path.join(args.save_root, args.base_model, args.task, f'FT-L{args.last_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')
    elif args.first_n is not None:
        save_dir = os.path.join(args.save_root, args.base_model, args.task, f'FT-F{args.first_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')
    else:
        save_dir = os.path.join(args.save_root, args.base_model, args.task,
                                f'FT-S{args.sample_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')

    os.makedirs(save_dir, exist_ok=True)

    if master_process:
        logging.info(args)

    args.save_dir = save_dir
    set_seed(args.seed)

    main(args)
