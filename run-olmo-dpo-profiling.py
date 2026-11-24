import argparse
import logging
import os

from types import SimpleNamespace

import torch
from torch import nn

from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, PreTrainedModel
from transformers import PreTrainedTokenizerBase, ProcessorMixin
from trl import DPOConfig

from utils.helper import naming_conversion
from utils.profiling_utils import (FeatureParameterDPOCollector, FeatureParameterDPOTrainer,
                                   LossSpaceDPOCollector, LossSpaceDPOLoggingCallback, RoundingScanLoggingCallback)

torch.backends.cuda.enable_cudnn_sdp(False)


class ProfilingDPOTrainer(FeatureParameterDPOTrainer):
    def _prepare_dataset(
            self,
            dataset: Dataset | IterableDataset,
            processing_class: PreTrainedTokenizerBase | ProcessorMixin,
            args: DPOConfig,
            dataset_name: str,
    ) -> Dataset | IterableDataset:
        stable_cache_args = SimpleNamespace(
            dataset_num_proc=args.dataset_num_proc,
            max_length=args.max_length,
            truncation_mode=args.truncation_mode,
            ld_alpha=args.ld_alpha,
        )
        return super()._prepare_dataset(
            dataset, processing_class, stable_cache_args, dataset_name
        )


def register_first_order_hook(model: nn.Module | PreTrainedModel | None,
                              collector: FeatureParameterDPOCollector,):
    collector.handles.append(
        model.model.layers[0].register_forward_pre_hook(
            collector.make_forward_hook(),
            with_kwargs=True,
        )
    )

    return collector


def main(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              padding_side="right",
                                              truncation_side='right')

    if 'Llama' in args.base_model:
        template_path = os.path.join('chat_templates', 'llama_template.jinja')
        with open(template_path, 'r') as f:
            tokenizer.chat_template = f.read()

    # Load dataset
    dataset = load_dataset(f"trl-lib/{args.task}")

    # create model
    model_dtype = torch.bfloat16 if args.data_type == 'bf16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 dtype=model_dtype,
                                                 attn_implementation='sdpa')
    model.config.use_cache = False

    model.requires_grad_(False)

    model_n_layer = model.config.num_hidden_layers
    assert 0 < args.last_n <= model_n_layer

    selected_layers = list(range(model_n_layer - args.last_n, model_n_layer))
    prefix_list = [f"model.layers.{l_idx}." for l_idx in selected_layers]

    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefix_list):
            p.requires_grad_(True)

    # training
    dpo_config = DPOConfig(output_dir=args.save_dir,
                           per_device_train_batch_size=args.batch_size,
                           per_device_eval_batch_size=args.eval_batch_size,
                           gradient_accumulation_steps=args.grad_accum,
                           learning_rate=args.learning_rate,
                           weight_decay=args.weight_decay,
                           adam_beta1=args.adam_beta1,
                           adam_beta2=args.adam_beta2,
                           max_grad_norm=args.max_grad_norm,
                           num_train_epochs=args.epochs,
                           warmup_steps=args.warmup_steps,
                           lr_scheduler_type=args.lr_scheduler_type,
                           lr_scheduler_kwargs=args.lr_scheduler_kwargs,
                           save_strategy='no',
                           bf16=True,
                           tf32=True,
                           gradient_checkpointing=args.enable_gradient_checkpointing,
                           gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
                           dataloader_pin_memory=True,
                           dataloader_persistent_workers=True,
                           dataloader_num_workers=1,
                           dataloader_prefetch_factor=4,
                           logging_strategy='steps',
                           logging_steps=args.logging_steps,
                           eval_strategy='no',
                           disable_tqdm=False,
                           seed=args.seed,
                           dataset_num_proc=2,
                           max_length=args.max_seq_length,
                           pad_to_multiple_of=8,
                           precompute_ref_log_probs=True,
                           precompute_ref_batch_size=args.ref_batch_size,
                           loss_type=args.loss_type,
                           beta=args.beta,)

    save_pth = os.path.join(args.save_dir, f'{args.profile_type}-{args.data_type}_record.pt')

    if args.profile_type == 'loss-space-update':
        collector = LossSpaceDPOCollector(profile_batch_size=None)
    else:
        collector = FeatureParameterDPOCollector(profile_batch_size=None)

    dpo_trainer = ProfilingDPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset['train'],
        processing_class=tokenizer,
        collector=collector,
    )
    assert dpo_trainer.accelerator.num_processes == 1, "Only support a single GPU."

    if args.profile_type == 'rounding-scan':
        hooking_cb = RoundingScanLoggingCallback(
            collector=collector,
            measure_every=args.measure_every,
            scan_exponents=args.scan_exponents,
            max_steps=args.max_steps,
            save_path=save_pth,
        )
    else:
        assert args.loss_type == 'sigmoid', "Loss-space profiling only supports the sigmoid DPO loss."
        register_first_order_hook(dpo_trainer.model,
                                  collector,)
        hooking_cb = LossSpaceDPOLoggingCallback(
            collector=collector,
            output_embeddings=dpo_trainer.model.get_output_embeddings(),
            autocast_context=dpo_trainer.accelerator.autocast,
            max_steps=args.max_steps,
            log_every=args.logging_steps,
            save_path=save_pth,
            chunk_size=args.jvp_chunk_size,
            token_chunk_size=args.token_chunk_size,
            beta=args.beta,
        )
    dpo_trainer.add_callback(hooking_cb)

    dpo_trainer.train()

    tmp_path = f"{save_pth}.tmp"
    torch.save({"logging_records": hooking_cb.logging_records,}, tmp_path)
    os.replace(tmp_path, save_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['Qwen3-0.6B', 'Llama-3.1-8B-Instruct', 'Llama-3.2-1B-Instruct'],
                        default='Qwen3-0.6B')
    parser.add_argument('--data-type', type=str,
                        choices=['bf16', 'fp32'],
                        default='fp32')
    parser.add_argument('--task', type=str,
                        choices=['ultrafeedback_binarized'],
                        default='ultrafeedback_binarized')

    # finetuning relevant
    parser.add_argument('--last-n', type=int, required=True)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--loss-type', type=str, default='sigmoid')

    # training relevant
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--ref-batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.999)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--warmup-steps', type=float, default=0.0)
    parser.add_argument('--lr-scheduler-type', type=str, default='constant')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--jvp-chunk-size', type=int, default=8)
    parser.add_argument('--token-chunk-size', type=int, default=32)
    parser.add_argument('--measure-every', type=int, default=500)
    parser.add_argument('--scan-exponents', type=float, nargs='+', default=list(range(8, 27)))
    parser.add_argument('--enable-gradient-checkpointing', action='store_true')

    # logging
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profile-type', type=str,
                        choices=['loss-space-update', 'rounding-scan'], required=True)

    args = parser.parse_args()

    if 'Llama' in args.base_model:
        args.model_name_or_path = f"meta-llama/{args.base_model}"
    else:
        args.model_name_or_path = f"Qwen/{args.base_model}"

    args.learning_rate = 2 ** (-args.learning_rate_exponent)

    args.suffix = '-constant' if args.lr_scheduler_type == 'constant' else ''
    args.suffix += '-bf16' if args.data_type == 'bf16' else ''

    if args.lr_scheduler_type == 'cosine_with_min_lr':
        args.lr_scheduler_kwargs = {'min_lr_rate': 0.10}
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
