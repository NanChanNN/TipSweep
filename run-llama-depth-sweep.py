import argparse
import logging
import os
import random

import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from utils.helper import naming_conversion, random_split
from utils.data_utils import filter_by_length
from utils.optim_utils import FP32StateAdamW

torch.backends.cuda.enable_cudnn_sdp(False)

def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{args.base_model}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_path = os.path.join('chat_templates', 'llama_template.jinja')
    with open(template_path, 'r') as f:
        tokenizer.chat_template = f.read()

    # Load dataset
    raw_data = load_dataset(f"allenai/{args.task}", split='train')

    if args.filter_length:
        raw_data = filter_by_length(raw_data, tokenizer, max_len=args.max_seq_length)

    train_indices, test_indices = random_split(len(raw_data), seed=args.seed)
    train_set = raw_data.select(train_indices[: 32 * args.train_set_size])
    test_set = raw_data.select(test_indices[: 32 * args.test_set_size])

    # create model
    model_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.data_type]
    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{args.base_model}",
                                                 dtype=model_dtype,)
    model.requires_grad_(False)

    model_n_layer = model.config.num_hidden_layers
    if args.selected_layer is not None:
        assert 0 <= args.selected_layer < model_n_layer
        prefix_list = [f"model.layers.{args.selected_layer}."]
    elif args.last_n is not None:
        assert 0 < args.last_n <= model_n_layer
        prefix_list = [f"model.layers.{l_idx}." for l_idx in range(model_n_layer - args.last_n, model_n_layer)]
    elif args.first_n is not None:
        assert 0 < args.first_n <= model_n_layer
        prefix_list = [f"model.layers.{l_idx}." for l_idx in range(args.first_n)]
    else:
        num_ft_layers = args.sample_n_layers
        assert 0 < num_ft_layers <= model_n_layer

        tmp_rng = random.Random(args.sample_n_seed)
        selected_layers = tmp_rng.sample(range(model_n_layer), num_ft_layers)
        prefix_list = [f"model.layers.{l_idx}." for l_idx in selected_layers]

        logging.info(f"selected layers: {selected_layers}")

    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefix_list):
            p.requires_grad_(True)

    # training
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
                           save_strategy='epoch',
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
                           eval_strategy='steps',
                           eval_steps=args.eval_steps,
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=16,
                           max_length=args.max_seq_length,
                           pad_to_multiple_of=8,
                           assistant_only_loss=True,)

    # set up optimizer
    optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(sft_config)
    if args.fp32_optim_states:
        optimizer_cls = FP32StateAdamW

    sft_trainer = SFTTrainer(model=model,
                             args=sft_config,
                             train_dataset=train_set,
                             eval_dataset=test_set,
                             processing_class=tokenizer,
                             optimizer_cls_and_kwargs=(optimizer_cls, optimizer_kwargs),)

    sft_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Llama-3.2-1B', 'Llama-3.1-8B'],
                        default='Llama-3.2-1B')
    parser.add_argument('--data-type', type=str,
                        choices=['bf16', 'fp32', 'fp16'],
                        default='bf16')
    parser.add_argument('--fp32-optim-states', action='store_true')
    parser.add_argument('--task', type=str,
                        choices=['tulu-3-sft-mixture'],
                        default='tulu-3-sft-mixture')

    # finetuning relevant
    ft_group = parser.add_mutually_exclusive_group(required=True)
    ft_group.add_argument('--selected-layer', type=int)
    ft_group.add_argument('--last-n', type=int)
    ft_group.add_argument('--first-n', type=int)
    ft_group.add_argument('--sample-n', type=str)  # format: number of layers-S{seed}

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--lr-scheduler-type', type=str,
                        choices=['constant', 'cosine_with_min_lr'],
                        default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--logging-steps', type=int, default=25)

    # training set size
    parser.add_argument('--train-set-size', type=int, default=10000)
    parser.add_argument('--test-set-size', type=int, default=1000)
    parser.add_argument('--filter-length', action='store_true')
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # logging
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.sample_n is not None:
        sample_n_parts = args.sample_n.split('-S')
        if len(sample_n_parts) != 2 or not sample_n_parts[0].isdigit() or not sample_n_parts[1].isdigit():
            parser.error("--sample-n must use format {num_layers}-S{seed}, e.g. 4-S42")
        args.sample_n_layers = int(sample_n_parts[0])
        args.sample_n_seed = int(sample_n_parts[1])

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = ''

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    elif args.lr_scheduler_type == 'constant':
        args.lr_scheduler_kwargs = {}
        args.suffix += '-constant'
    else:
        args.lr_scheduler_kwargs = {}

    args.suffix += f'-{args.data_type}'

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}

    if args.selected_layer is not None:
        save_dir = os.path.join('checkpoints', args.base_model, args.task, f'FT-{args.selected_layer}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')
    elif args.last_n is not None:
        save_dir = os.path.join('checkpoints', args.base_model, args.task, f'FT-L{args.last_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')
    elif args.first_n is not None:
        save_dir = os.path.join('checkpoints', args.base_model, args.task, f'FT-F{args.first_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')
    else:
        save_dir = os.path.join('checkpoints', args.base_model, args.task,
                                f'FT-S{args.sample_n}{args.suffix}',
                                f'LR-{naming_conversion(args.learning_rate_exponent)}')

    os.makedirs(save_dir, exist_ok=True)

    logging.info(args)

    args.save_dir = save_dir
    set_seed(args.seed)

    main(args)
