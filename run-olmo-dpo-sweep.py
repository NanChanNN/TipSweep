import argparse
import logging
import os
import random
import numpy as np
from functools import partial
from types import SimpleNamespace

import torch
from datasets import Dataset, IterableDataset

from transformers import AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin
from datasets import load_dataset

from utils.helper import naming_conversion
from utils.general_utils import configure_trainable_layers
from trl import DPOConfig, DPOTrainer
from trl.trainer.utils import pad

torch.backends.cuda.enable_cudnn_sdp(False)

global_rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
master_process = global_rank == 0


def tokenize_olmo_preferences(example, tokenizer, max_length):
    tokenized = tokenizer.apply_chat_template([example['chosen'], example['rejected']],
                                              truncation=True,
                                              max_length=max_length,
                                              return_dict=True,
                                              return_assistant_tokens_mask=True)
    return {f'{key}_{field}': tokenized[field][i]
            for i, key in enumerate(['chosen', 'rejected'])
            for field in ['input_ids', 'assistant_masks']}


def olmo_dpo_collate_fn(examples, pad_token_id):
    input_ids = [torch.tensor(example[f'{key}_input_ids'])
                 for key in ['chosen', 'rejected'] for example in examples]
    completion_mask = [torch.tensor(example[f'{key}_assistant_masks'])
                       for key in ['chosen', 'rejected'] for example in examples]
    batch = {
        'input_ids': pad(input_ids, padding_value=pad_token_id, pad_to_multiple_of=8),
        'attention_mask': pad([torch.ones_like(ids) for ids in input_ids], pad_to_multiple_of=8),
        'completion_mask': pad(completion_mask, pad_to_multiple_of=8),
    }
    for key in ['ref_chosen_logps', 'ref_rejected_logps']:
        if key in examples[0]:
            batch[key] = torch.tensor([example[key] for example in examples])
    return batch


class StableRefCacheDPOTrainer(DPOTrainer):
    def _prepare_dataset(
            self,
            dataset: Dataset | IterableDataset,
            processing_class: PreTrainedTokenizerBase | ProcessorMixin,
            args: DPOConfig,
            dataset_name: str,
        ) -> Dataset | IterableDataset:
        if 'OLMo' in processing_class.name_or_path:
            with args.main_process_first():
                dataset = dataset.map(tokenize_olmo_preferences,
                                       fn_kwargs={'tokenizer': processing_class, 'max_length': args.max_length},
                                       num_proc=args.dataset_num_proc,
                                       remove_columns=dataset.column_names)
                return dataset.filter(lambda example: any(example['chosen_assistant_masks'])
                                                      and any(example['rejected_assistant_masks']),
                                       num_proc=args.dataset_num_proc)

        stable_cache_args = SimpleNamespace(
            dataset_num_proc=args.dataset_num_proc,
            max_length=args.max_length,
            truncation_mode=args.truncation_mode,
            ld_alpha=args.ld_alpha,
        )
        return super()._prepare_dataset(
            dataset, processing_class, stable_cache_args, dataset_name
        )


def main(args):
    # load tokenizer 
    is_olmo = 'OLMo' in args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              padding_side="right",
                                              truncation_side='right')

    if 'Llama' in args.base_model or 'OLMo' in args.base_model:
        template_name = 'llama' if 'Llama' in args.base_model else args.base_model
        template_path = os.path.join('chat_templates', f'{template_name}_template.jinja')
        with open(template_path, 'r') as f:
            tokenizer.chat_template = f.read()
    
    # Load training dataset
    if args.task == 'olmo-2-1124-7b-preference-mix':
        dataset = load_dataset(f"allenai/{args.task}",
                               split='train')
        dataset = dataset.select_columns(['chosen', 'rejected'])

        filtered_id_path = os.path.join('auxiliary_data', f'{args.task}_filtered_ids.npz')
        with np.load(filtered_id_path) as data:
            keep_idx = np.setdiff1d(np.arange(len(dataset)), data['filtered_ids'])
        dataset = dataset.select(keep_idx)
        dataset = dataset.train_test_split(test_size=args.test_set_size,
                                            train_size=args.train_set_size,
                                            seed=args.seed + 11)
    else:
        dataset = load_dataset(f"trl-lib/{args.task}",)
                
    # Setup Config
    dpo_config = DPOConfig(output_dir=args.save_dir,
                           model_init_kwargs=args.model_init_kwargs,
                           per_device_train_batch_size=args.per_device_batch_size,
                           per_device_eval_batch_size=args.per_device_eval_batch_size,
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
                           deepspeed=args.deepspeed_config_file,
                           gradient_checkpointing=args.enable_gradient_checkpointing,
                           gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=1,
                           dataloader_prefetch_factor=4,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='steps',
                           eval_steps=args.eval_steps,
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=2,
                           remove_unused_columns=not is_olmo,
                           max_length=args.max_seq_length,
                           pad_to_multiple_of=8,
                           precompute_ref_log_probs=True,
                           precompute_ref_batch_size=args.per_device_ref_batch_size,
                           loss_type=args.loss_type,
                           beta=args.beta,)

    # Configure Trainer
    trainer = StableRefCacheDPOTrainer(
        model=args.model_name_or_path,
        args=dpo_config,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=partial(olmo_dpo_collate_fn, pad_token_id=tokenizer.pad_token_id) if is_olmo else None,
    )

    # Setup trainable layers
    configure_trainable_layers(model=trainer.model,
                               last_n=args.last_n)
    trainer.model.config.use_cache = False
    
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # basic setting
    parser.add_argument('--base-model', type=str,
                        choices=['Qwen3-0.6B', 'Llama-3.1-8B-Instruct', 'Llama-3.2-1B-Instruct',
                                 'OLMo-2-1124-7B-SFT'],
                        default='Qwen3-0.6B')
    parser.add_argument('--data-type', type=str,
                        choices=['bf16', 'fp32'],
                        default='fp32')
    parser.add_argument('--task', type=str,
                        choices=['ultrafeedback_binarized', 'olmo-2-1124-7b-preference-mix'],
                        default='ultrafeedback_binarized')
    
    # finetuning related
    parser.add_argument('--last-n', type=int, required=True)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    
    # training related
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--per-device-batch-size', type=int, default=8)
    parser.add_argument('--per-device-eval-batch-size', type=int, default=16)
    parser.add_argument('--per-device-ref-batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=None)
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')
    parser.add_argument('--deepspeed-config-file', type=str, default=None)
    parser.add_argument('--train-set-size', type=int, default=24000)
    parser.add_argument('--test-set-size', type=int, default=2400)
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--loss-type', type=str, default='sigmoid')
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--save-root', type=str, default='checkpoints')
    parser.add_argument('--save-checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    if 'Llama' in args.base_model:
        args.model_name_or_path = f"meta-llama/{args.base_model}"
    elif 'Qwen' in args.base_model:
        args.model_name_or_path = f"Qwen/{args.base_model}"
    elif 'OLMo' in args.base_model:
        args.model_name_or_path = f"allenai/{args.base_model}"
        if args.data_type == 'fp32' and args.deepspeed_config_file is None:
            args.deepspeed_config_file = 'configs/ds_configs/stage3_no_offloading_accelerate.conf'
    else:
        raise ValueError("Unsupported model!")

    args.model_init_kwargs = {'dtype': 'bfloat16' if args.data_type == 'bf16' else 'float32'}

    if args.max_seq_length is None:
        args.max_seq_length = 2048 if args.task == 'olmo-2-1124-7b-preference-mix' else 1024
    
    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.10}
    else:
        args.lr_scheduler_kwargs = {}

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}

    args.suffix = f'-{args.data_type}'
    save_dir = os.path.join(args.save_root, args.base_model, args.task, 
                            f'FT-L{args.last_n}{args.suffix}',
                            f'LR-{naming_conversion(args.learning_rate_exponent)}')

    if master_process:
        logging.info(args)
        
        effective_batch_size = args.per_device_batch_size * world_size * args.grad_accum
        logging.info(f"Effective batch size is {effective_batch_size}")

    args.save_dir = save_dir
    
    # set random seed
    random.seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    torch.manual_seed(args.seed + global_rank)
    torch.cuda.manual_seed_all(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    main(args)
