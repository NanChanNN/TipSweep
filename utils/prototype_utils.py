import torch
from typing import List


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
