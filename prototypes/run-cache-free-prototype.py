import argparse
import copy
import datetime
import logging
import os
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

from functools import partial
from typing import Optional, Tuple

import math
import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from utils.data_utils import code_prompt_completion_converter
from utils.file_utils import load_filtered_ids
from utils.prototype_utils import tokenize_fn, prompt_completion_collate_fn, get_parameter_groups, lr_mult

logger = logging.getLogger(__name__)


def custom_split(model,
                 output_layer: int = 0,
                 grid_size: int = 1,
                 upcast: bool = False) -> Tuple:
    # forward only modules
    fwd_dict = nn.ModuleDict()
    fwd_dict['embed_tokens'] = model.model.embed_tokens
    fwd_dict['rotary_emb'] = model.model.rotary_emb
    fwd_dict['model_norm'] = model.model.norm
    fwd_dict['lm_head'] = model.lm_head

    for l_idx in range(output_layer):
        fwd_dict[f'layers_{l_idx}'] = model.model.layers[l_idx]

    fwd_dict.requires_grad_(False)

    # backward only modules
    suffix = nn.ModuleList()
    for l_idx in range(output_layer, model.config.num_hidden_layers):
        suffix.append(model.model.layers[l_idx])
    suffix.requires_grad_(True)

    if upcast:
        suffix.to(dtype=torch.float32)

    fwd_bwd_list = [suffix]
    for _ in range(1, grid_size):
        fwd_bwd_list.append(copy.deepcopy(suffix))

    return fwd_dict, fwd_bwd_list


def custom_forward_backward(fwd_dict,
                            fwd_bwd_list,
                            config,
                            output_layer: int,
                            loss_fn,
                            input_ids: Optional[torch.LongTensor] = None,
                            attention_mask: Optional[torch.Tensor] = None,
                            labels: Optional[torch.Tensor] = None,
                            num_items_in_batch: int = None,
                            upcast: bool = False,):
    # shift the label
    shift_labels = labels[..., 1:]
    shift_mask = shift_labels != -100
    batch_indices, token_indices = shift_mask.nonzero(as_tuple=True)
    shift_labels = shift_labels[batch_indices, token_indices]
    loss_scale = shift_mask.sum() / num_items_in_batch

    # input embedding
    inputs_embeds: torch.Tensor = fwd_dict['embed_tokens'](input_ids)

    # position embedding
    cache_position: torch.Tensor = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = fwd_dict['rotary_emb'](inputs_embeds, position_ids)

    # causal mask
    causal_mask = create_causal_mask(config=config,
                                     inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     past_key_values=None,
                                     position_ids=position_ids,)

    # prefix forwarding
    hidden_states = inputs_embeds
    for l_idx in range(output_layer):
        hidden_states = fwd_dict[f'layers_{l_idx}'](hidden_states,
                                                    attention_mask=causal_mask,
                                                    position_ids=position_ids,
                                                    past_key_values=None,
                                                    cache_position=cache_position,
                                                    position_embeddings=position_embeddings,)

    # suffix forwarding
    losses = torch.empty((len(fwd_bwd_list)), dtype=torch.float32, device=hidden_states.device)
    for s_idx, suffix in enumerate(fwd_bwd_list):
        hidden_states_s = hidden_states

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=upcast):
            for i in range(len(suffix)):
                hidden_states_s = suffix[i](hidden_states_s,
                                            attention_mask=causal_mask,
                                            position_ids=position_ids,
                                            past_key_values=None,
                                            cache_position=cache_position,
                                            position_embeddings=position_embeddings,)
            hidden_states_s = hidden_states_s[batch_indices, token_indices]
            hidden_states_s = fwd_dict['model_norm'](hidden_states_s)

            loss_s = loss_fn(fwd_dict['lm_head'].weight, hidden_states_s, shift_labels)
            loss_s *= loss_scale
        loss_s.backward()

        losses[s_idx] = loss_s.detach()
    return losses


def main(args):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{args.base_model}")

    # load dataset
    raw_data = load_dataset(f"ise-uiuc/{args.task}", split="train")

    if args.filter_length:
        tmp_id_path = os.path.join('auxiliary_data', f'{args.task}_{args.base_model}_filtered_ids.npz')
        filtered_train_ids = load_filtered_ids(tmp_id_path)
        raw_data = raw_data.filter(lambda _, idx: idx not in filtered_train_ids, with_indices=True)

    # convert raw data into formatted data
    formatted_trainset = raw_data.map(code_prompt_completion_converter,
                                      batched=True,
                                      remove_columns=list(raw_data.column_names))

    # tokenize the formatted data and generate the completion mask
    tokenized_dataset = formatted_trainset.map(tokenize_fn,
                                               fn_kwargs={"processing_class": tokenizer},
                                               num_proc=16,
                                               remove_columns=list(formatted_trainset.column_names))

    # build data loader
    my_collate_fn = partial(prompt_completion_collate_fn,
                            pad_token_id=tokenizer.pad_token_id,
                            pad_to_multiple_of=8)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(tokenized_dataset,
                              collate_fn=my_collate_fn,
                              batch_size=args.batch_size,
                              num_workers=4,
                              prefetch_factor=2,
                              pin_memory=True,
                              drop_last=False,
                              sampler=RandomSampler(tokenized_dataset, generator=sampler_generator))

    # load model
    model = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{args.base_model}",
                                                 dtype=torch.bfloat16,
                                                 attn_implementation=args.attn_impl)
    assert args.output_layer < model.config.num_hidden_layers, (
        f"--output-layer must be < {model.config.num_hidden_layers}, got {args.output_layer}"
    )

    model.to(args.device)
    model.train()
    fwd_dict, fwd_bwd_list = custom_split(model,
                                          output_layer=args.output_layer,
                                          grid_size=len(args.candidate_learning_rates),
                                          upcast=args.upcast)

    # create optimizer
    total_group_parameters, trainable_params_list = [], []
    for m_idx, module in enumerate(fwd_bwd_list):
        grouped_parameters, trainable_params = get_parameter_groups(module,
                                                                    learning_rate=args.candidate_learning_rates[m_idx],
                                                                    weight_decay=args.weight_decay,)

        total_group_parameters.extend(grouped_parameters)
        trainable_params_list.append(trainable_params)

    optimizer = AdamW(total_group_parameters,
                      betas=(0.9, 0.999),
                      fused=True)

    # build lr scheduler
    total_steps = math.ceil(len(train_loader) / args.grad_accum)
    max_train_batches = len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_lambda = partial(lr_mult,
                        warmup_steps=warmup_steps,
                        total_steps=total_steps,)

    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lr_lambda)

    # loss function
    loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="mean")

    # training
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    batches_seen = 0
    ema_losses = None
    start_time = time.perf_counter()
    for step in range(total_steps):
        current_accum_steps = min(args.grad_accum, max_train_batches - batches_seen)
        batch_samples = [next(train_iter) for _ in range(current_accum_steps)]
        num_items_in_batch = sum(batch["labels"].ne(-100).sum() for batch in batch_samples).to(args.device)

        step_losses = torch.zeros(len(fwd_bwd_list), device=args.device)
        for batch in batch_samples:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}

            step_losses += custom_forward_backward(fwd_dict,
                                                   fwd_bwd_list,
                                                   config=model.config,
                                                   output_layer=args.output_layer,
                                                   loss_fn=loss_fn,
                                                   input_ids=batch['input_ids'],
                                                   attention_mask=batch['attention_mask'],
                                                   labels=batch['labels'],
                                                   num_items_in_batch=num_items_in_batch,
                                                   upcast=args.upcast,)
            batches_seen += 1

        if args.max_grad_norm > 0:
            for trainable_params in trainable_params_list:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)

        optimizer.step()

        ema_losses = step_losses if ema_losses is None else 0.1 * step_losses + 0.9 * ema_losses
        if (step + 1) % args.logging_steps == 0 or step + 1 == total_steps:
            ema_text = ", ".join(f"2^-{exponent:g}: {ema_loss:.4f}"
                                 for exponent, ema_loss in zip(args.candidate_exponents, ema_losses.tolist()))
            logger.info(f"step {step + 1}/{total_steps} | lr scale {lr_lambda(step):.3f} | EMA loss {ema_text}")

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    training_time = datetime.timedelta(seconds=round(time.perf_counter() - start_time))
    logger.info(f"finished {total_steps} steps | training time {training_time} | "
                f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")

    final_losses = ema_losses.tolist()
    best_idx = min(range(len(final_losses)),
                   key=lambda i: final_losses[i] if math.isfinite(final_losses[i]) else math.inf)
    logger.info("final EMA training loss per candidate:")
    for c_idx, (exponent, ema_loss) in enumerate(zip(args.candidate_exponents, final_losses)):
        mark = "  (best)" if c_idx == best_idx and math.isfinite(ema_loss) else ""
        logger.info(f"  lr = 2^-{exponent:g}: {ema_loss:.4f}{mark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic arguments
    parser.add_argument('--base-model', type=str,
                        choices=['deepseek-coder-6.7b-base'],
                        default='deepseek-coder-6.7b-base')
    parser.add_argument('--task', type=str,
                        choices=['Magicoder-OSS-Instruct-75K'],
                        default='Magicoder-OSS-Instruct-75K')
    parser.add_argument('--attn-impl', type=str,
                        choices=['flash_attention_2', 'eager'],
                        default='flash_attention_2')
    parser.add_argument('--filter-length', action='store_true')

    # extraction relevant
    parser.add_argument('--output-layer', type=int, default=31)

    # training relevant
    parser.add_argument('--precision', type=str,
                        choices=['bf16', 'fp32'],
                        default='bf16' ,)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--candidate-exponents', type=float, nargs='+', default=list(range(11, 15)))
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)

    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging-steps', type=int, default=10)

    args = parser.parse_args()

    assert args.output_layer > 0, "--output-layer should be positive"
    assert args.logging_steps > 0, "--logging-steps should be positive"

    args.device = torch.device('cuda')
    args.candidate_learning_rates = [2 ** (-item) for item in args.candidate_exponents]
    args.upcast = args.precision == 'fp32'

    # print this script's messages to stdout and keep library messages at the warning level
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
    logger.setLevel(logging.INFO)
    logger.info(args)

    set_seed(args.seed)

    main(args)
