import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import set_seed, AutoModelForCausalLM
from transformers.trainer_pt_utils import get_parameter_names
from transformers.masking_utils import create_causal_mask

from safetensors import safe_open
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def pad_fn_nd(tensor_list: List[torch.Tensor],
           padding_value: int = 0,
           padding_side: str = "right",
           pad_to_multiple_of: int | None = None,) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensor_list)
    if pad_to_multiple_of is not None:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    output = tensor_list[0].new_full((len(tensor_list), max_len, *tensor_list[0].shape[1:]), 
                                     padding_value)

    if padding_side == "right":
        for i, t in enumerate(tensor_list):
            output[i, :t.size(0)] = t
    elif padding_side == "left":
        for i, t in enumerate(tensor_list):
            output[i, max_len - t.size(0):] = t
    else:
        raise ValueError("padding_side must be 'left' or 'right'")

    return output


def custom_collate_fn(batch,):
    hidden_states = [example['feats'] for example in batch]
    labels = [example['labels'] for example in batch]
    attention_mask = [torch.ones_like(x) for x in labels]

    output = {}
    output['hidden_states'] = pad_fn_nd(hidden_states,
                                        padding_value=0,
                                        padding_side='right',
                                        pad_to_multiple_of=8)
    output['labels'] = pad_fn_nd(labels, 
                                 padding_value=-100,
                                 padding_side='right',
                                 pad_to_multiple_of=8)
    output['attention_mask'] = pad_fn_nd(attention_mask,
                                         padding_value=0,
                                         padding_side='right',
                                         pad_to_multiple_of=8)

    return output


def read_feature_batch(path: str | Path,
                       device: str | torch.device = "cpu"
                       ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    data = load_file(str(path), device=str(device))

    feats = data["feats"]
    labels = data["labels"]
    offsets = data["offsets"].tolist()

    feat_list = [feats[s:e] for s, e in zip(offsets[:-1], offsets[1:])]
    label_list = [labels[s:e] for s, e in zip(offsets[:-1], offsets[1:])]

    return feat_list, label_list


class FeatureCacheDataset(IterableDataset):
    """Streams the samples cached by run-prototype-s1.py, one safetensors file at a time."""

    def __init__(self, cache_dir: str | Path, repeat: int = 1, seed: int = 0):
        self.files = sorted(Path(cache_dir).glob('batch_*.safetensors'))
        assert len(self.files) > 0, f"no cached features found in {cache_dir}"
        self.num_samples = sum(self._num_samples(f) for f in self.files)
        self.repeat = repeat
        self.seed = seed

    @staticmethod
    def _num_samples(path: Path) -> int:
        with safe_open(str(path), framework="pt") as f:
            return int(f.metadata()["num_samples"])

    def __len__(self):
        return self.num_samples * self.repeat

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.repeat):
            for f_idx in torch.randperm(len(self.files), generator=generator).tolist():
                feat_list, label_list = read_feature_batch(self.files[f_idx])
                for feats, labels in zip(feat_list, label_list):
                    yield {"feats": feats, "labels": labels}


def custom_configure(model, 
                     torch_device, 
                     output_layer: int=0,
                     precision: str='bf16'):
    model_n_layers = model.config.num_hidden_layers
    
    model.requires_grad_(False)
    
    model.model.rotary_emb.to(torch_device)
    
    for l_idx in range(output_layer, model_n_layers):
        model.model.layers[l_idx].requires_grad_(True)
        model.model.layers[l_idx].to(torch_device)
        if precision == 'fp32':
            # fp32 master weights (hence fp32 gradients and optimizer states); compute stays bf16 via autocast
            model.model.layers[l_idx].float()

    model.model.norm.to(torch_device)
    model.lm_head.to(torch_device)


def custom_forward_2(model, 
                     hidden_states: Optional[torch.Tensor] = None,
                     attention_mask: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None, 
                     output_layer: int = 0,
                     num_items_in_batch: int = None):
    # position embedding
    cache_position: torch.Tensor = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    
    # causal mask
    causal_mask = create_causal_mask(config=model.model.config, 
                                     inputs_embeds=hidden_states,
                                     attention_mask=attention_mask,
                                     past_key_values=None,
                                     position_ids=position_ids,)
    
    # continue forwarding
    for decoder_layer in model.model.layers[output_layer:]:
        hidden_states = decoder_layer(hidden_states,
                                      attention_mask=causal_mask,
                                      position_ids=position_ids,
                                      past_key_values=None,
                                      cache_position=cache_position,
                                      position_embeddings=position_embeddings,)    
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    
    loss = model.loss_function(logits=logits, 
                               labels=labels, 
                               vocab_size=model.config.vocab_size,
                               num_items_in_batch=num_items_in_batch)
    
    return loss


def main(args):
    # load cached features
    dataset = FeatureCacheDataset(args.cache_dir, seed=args.seed)
    
    train_loader = DataLoader(dataset,
                              collate_fn=custom_collate_fn,
                              batch_size=args.batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False)
    
    # load model
    model = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{args.base_model}",
                                                 dtype=torch.bfloat16,
                                                 attn_implementation=args.attn_impl)
    assert args.output_layer < model.config.num_hidden_layers, (
        f"--output-layer must be < {model.config.num_hidden_layers}, got {args.output_layer}"
    )
    custom_configure(model, args.device, args.output_layer, args.precision)

    # create optimizer
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        }
    ]
    trainable_params = [p for group in grouped_parameters for p in group["params"]]
    optimizer = AdamW(grouped_parameters,
                      lr=args.learning_rate,
                      betas=(0.9, 0.999),
                      fused=True)

    # build lr scheduler
    total_batches = len(train_loader)
    max_train_batches = total_batches
    total_steps = math.ceil(max_train_batches / args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_start_factor = 1e-04
    min_lr_mult = 0.1

    def lr_mult(step: int) -> float:
        if warmup_steps > 0 and step <= warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)

        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_mult + (1.0 - min_lr_mult) * cosine

    lr_scheduler = LambdaLR(optimizer,
                            lr_lambda=lr_mult)
    
    # training
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)
    batches_seen = 0
    ema_loss = None
    start_time = time.perf_counter()
    for step in range(total_steps):
        current_accum_steps = min(args.grad_accum, max_train_batches - batches_seen)
        batch_samples = [next(train_iter) for _ in range(current_accum_steps)]
        num_items_in_batch = sum(batch["labels"].ne(-100).sum() for batch in batch_samples).to(args.device)

        step_loss = torch.zeros((), device=args.device)
        for batch in batch_samples:
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'fp32'):
                loss = custom_forward_2(model, 
                                        hidden_states=batch['hidden_states'],
                                        attention_mask=batch['attention_mask'],
                                        labels=batch['labels'], 
                                        output_layer=args.output_layer,
                                        num_items_in_batch=num_items_in_batch)
            
            loss.backward()
            step_loss += loss.detach()
            batches_seen += 1

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
        optimizer.step()

        # exponential moving average of the training loss, as in the paper
        ema_loss = step_loss if ema_loss is None else 0.1 * step_loss + 0.9 * ema_loss
        if (step + 1) % args.logging_steps == 0 or step + 1 == total_steps:
            logger.info(f"step {step + 1}/{total_steps} | lr {lr_scheduler.get_last_lr()[0]:.3e} | "
                        f"EMA loss {ema_loss.item():.4f}")

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    training_time = datetime.timedelta(seconds=round(time.perf_counter() - start_time))
    logger.info(f"finished {total_steps} steps | training time {training_time} | "
                f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
    logger.info(f"final EMA training loss at lr = 2^-{args.learning_rate_exponent:g}: {ema_loss.item():.4f}")


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
                        default='bf16',)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.00)
    parser.add_argument('--learning-rate-exponent', type=float, required=True)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)

    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-root', type=str, default='checkpoints')
    parser.add_argument('--logging-steps', type=int, default=10)

    args = parser.parse_args()

    assert args.output_layer > 0, "--output-layer should be positive"
    assert args.logging_steps > 0, "--logging-steps should be positive"

    args.device = torch.device('cuda')
    args.learning_rate = 2 ** (-args.learning_rate_exponent)
    args.suffix = ''

    cache_name = 'feature_cache'
    args.cache_dir = os.path.join(args.cache_root, args.base_model, args.task, 
                                  f'FT-O{args.output_layer}{args.suffix}', 
                                  cache_name)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stdout)
    logger.setLevel(logging.INFO)
    logger.info(args)
    
    set_seed(args.seed)
    
    main(args)
