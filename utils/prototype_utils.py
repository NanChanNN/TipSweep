import torch
from typing import List
import math
import torch.nn as nn
from transformers.trainer_pt_utils import get_parameter_names


def tokenize_fn(example, processing_class):
    output = {}

    completion = example["completion"]
    if not completion.endswith(processing_class.eos_token):
        completion += processing_class.eos_token

    prompt_ids = processing_class(text=example["prompt"])["input_ids"]
    prompt_completion_ids = processing_class(
        text=example["prompt"] + completion,
    )["input_ids"]

    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))

    output['input_ids'] = prompt_completion_ids
    output['completion_mask'] = completion_mask
    return output


def pad_fn(tensor_list: List[torch.Tensor],
           padding_value: int = 0,
           padding_side: str = "right",
           pad_to_multiple_of: int | None = None,) -> torch.Tensor:
    max_len = max(t.size(0) for t in tensor_list)
    if pad_to_multiple_of is not None:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    output = tensor_list[0].new_full((len(tensor_list), max_len), padding_value)

    if padding_side == "right":
        for i, t in enumerate(tensor_list):
            output[i, :t.size(0)] = t
    elif padding_side == "left":
        for i, t in enumerate(tensor_list):
            output[i, max_len - t.size(0):] = t
    else:
        raise ValueError("padding_side must be 'left' or 'right'")

    return output


def prompt_completion_collate_fn(batch, 
                                 pad_token_id: int, 
                                 pad_to_multiple_of: int | None = None,):
    # convert to tensor
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    labels = [torch.tensor(example['input_ids']) for example in batch]
    completion_mask = [torch.tensor(example['completion_mask']) for example in batch]
    
    # create the attention mask
    attention_mask = [torch.ones_like(ids) for ids in input_ids]

    output = {}
    output['input_ids'] = pad_fn(input_ids,
                                 padding_value=pad_token_id,
                                 padding_side='right',
                                 pad_to_multiple_of=pad_to_multiple_of)
    
    output['labels'] = pad_fn(labels, 
                              padding_value=-100,
                              padding_side='right',
                              pad_to_multiple_of=pad_to_multiple_of)
    
    output['attention_mask'] = pad_fn(attention_mask,
                                      padding_value=0,
                                      padding_side='right',
                                      pad_to_multiple_of=pad_to_multiple_of)
    
    completion_mask = pad_fn(completion_mask,
                             padding_value=0,
                             padding_side='right',
                             pad_to_multiple_of=pad_to_multiple_of)
    output['labels'][completion_mask == 0] = -100
    
    return output


def get_parameter_groups(model: nn.Module, learning_rate: float, weight_decay: float,):
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)

    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        }
    ]

    trainable_params = [p for group in grouped_parameters for p in group["params"]]

    return grouped_parameters, trainable_params


def lr_mult(step: int,
            warmup_steps: int,
            total_steps: int,
            warmup_start_factor: float = 1e-04,
            min_lr_mult: float = 0.1) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return warmup_start_factor + (1.0 - warmup_start_factor) * (step / warmup_steps)

    denom = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / denom
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_mult + (1.0 - min_lr_mult) * cosine
