import argparse
import logging
import os

import torch
from torch import nn

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, PreTrainedModel
from trl import SFTConfig

from utils.helper import naming_conversion, random_split
from utils.data_utils import filter_by_length
from utils.profiling_utils import (FeatureParameterCollector, FeatureParameterSFTTrainer,
                                   LossSpaceCollector, LossSpaceUpdateLoggingCallback, RoundingScanLoggingCallback)


def register_first_order_hook(model: nn.Module | PreTrainedModel | None,
                              collector: FeatureParameterCollector,):
    collector.handles.append(
        model.model.layers[0].register_forward_pre_hook(
            collector.make_forward_hook(),
            with_kwargs=True,
        )
    )

    return collector


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

    train_indices, _ = random_split(len(raw_data), seed=args.seed)
    train_set = raw_data.select(train_indices[: 32 * args.train_set_size])
    test_set = None

    # create model
    if args.data_type == 'fp32':
        model_dtype = torch.float32
    else:
        model_dtype = torch.bfloat16
    
    attn_implementations = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{args.base_model}",
                                                 dtype=model_dtype,
                                                 attn_implementation=attn_implementations)

    if args.last_n == 0:
        model.requires_grad_(True)
    else:
        model.requires_grad_(False)

        model_n_layer = model.config.num_hidden_layers
        assert 0 < args.last_n <= model_n_layer

        selected_layers = list(range(model_n_layer - args.last_n, model_n_layer))
        prefix_list = [f"model.layers.{l_idx}." for l_idx in selected_layers]

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
                           save_strategy='no',
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
                           assistant_only_loss=True,)

    save_pth = os.path.join(args.save_dir, f'{args.profile_type}{args.suffix}_record.pt')


    if args.profile_type in ['loss-space-update']:
        collector = LossSpaceCollector(profile_batch_size=None)

        sft_trainer = FeatureParameterSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_set,
            eval_dataset=test_set,
            processing_class=tokenizer,
            collector=collector,
        )
        assert sft_trainer.accelerator.num_processes == 1, "Only support a single GPU."
        assert args.last_n > 0, "Only support a positive last-n."

        register_first_order_hook(sft_trainer.model,
                                  collector,)

        hooking_cb = LossSpaceUpdateLoggingCallback(
            collector=collector,
            output_embeddings=sft_trainer.model.get_output_embeddings(),
            autocast_context=sft_trainer.accelerator.autocast,
            max_steps=args.max_steps,
            log_every=args.logging_steps,
            save_path=save_pth,
            chunk_size=args.jvp_chunk_size,
            token_chunk_size=args.token_chunk_size,
        )
        sft_trainer.add_callback(hooking_cb)
    elif args.profile_type == 'rounding-scan':
        collector = FeatureParameterCollector(profile_batch_size=None)
        sft_trainer = FeatureParameterSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_set,
            eval_dataset=test_set,
            processing_class=tokenizer,
            collector=collector,
        )
        assert sft_trainer.accelerator.num_processes == 1, "Only support a single GPU."
        assert args.last_n > 0, "Only support a positive last-n."

        hooking_cb = RoundingScanLoggingCallback(
            collector=collector,
            measure_every=args.measure_every,
            scan_exponents=args.scan_exponents,
            max_steps=args.max_steps,
            save_path=save_pth,
        )
        sft_trainer.add_callback(hooking_cb)
    else:
        raise ValueError

    sft_trainer.train()

    tmp_path = f"{save_pth}.tmp"
    torch.save({"logging_records": hooking_cb.logging_records,}, tmp_path)
    os.replace(tmp_path, save_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Llama-3.2-1B', 'Llama-3.1-8B'],
                        default='Llama-3.2-1B')
    parser.add_argument('--data-type', type=str,
                        choices=['bf16', 'fp32'],
                        default='bf16')
    parser.add_argument('--task', type=str,
                        choices=['tulu-3-sft-mixture'],
                        default='tulu-3-sft-mixture')
    

    # finetuning relevant
    parser.add_argument('--last-n', type=int, required=True)

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
    parser.add_argument('--lr-scheduler-type', type=str, default='cosine_with_min_lr')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--jvp-chunk-size', type=int, default=4)
    parser.add_argument('--token-chunk-size', type=int, default=32)
    parser.add_argument('--measure-every', type=int, default=500)
    parser.add_argument('--scan-exponents', type=float, nargs='+', default=list(range(8, 27)))

    # training set size
    parser.add_argument('--train-set-size', type=int, default=10000)
    parser.add_argument('--filter-length', action='store_true')
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # logging
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile-type', type=str,
                        choices=['loss-space-update', 'rounding-scan'], required=True)

    args = parser.parse_args()

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = ''

    if args.data_type == 'fp32':
        args.suffix += '-fp32'

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.1}
    else:
        args.lr_scheduler_kwargs = {}

    if args.enable_gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        args.gradient_checkpointing_kwargs = {}

    save_dir = os.path.join('profiling_records', args.base_model, args.task,
                            f'{args.profile_type}',
                            f'FT-L{args.last_n}',
                            f'LR-{naming_conversion(args.learning_rate_exponent)}')

    os.makedirs(save_dir, exist_ok=True)

    logging.info(args)

    args.save_dir = save_dir
    set_seed(args.seed, deterministic=True)

    main(args)
