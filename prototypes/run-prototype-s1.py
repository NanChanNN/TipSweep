import argparse
import logging
import os
from typing import Optional
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from datasets import tqdm
from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask

from utils.data_utils import code_prompt_completion_converter
from utils.prototype_utils import tokenize_fn, prompt_completion_collate_fn
from utils.file_utils import load_filtered_ids


def custom_forward(decoder, 
                   input_ids: Optional[torch.LongTensor] = None,
                   attention_mask: Optional[torch.Tensor] = None, 
                   output_layer: int = 0):
    end_l_idx = output_layer if output_layer != 0 else decoder.config.num_hidden_layers
    
    # input embedding
    inputs_embeds: torch.Tensor = decoder.embed_tokens(input_ids)
    
    # position embedding
    cache_position: torch.Tensor = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = decoder.rotary_emb(inputs_embeds, position_ids)
    
    # causal mask
    causal_mask = create_causal_mask(config=decoder.config, 
                                     inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     past_key_values=None,
                                     position_ids=position_ids,)
    
    # attention blocks
    hidden_states = inputs_embeds
    for decoder_layer in decoder.layers[: end_l_idx]:
        hidden_states = decoder_layer(hidden_states,
                                      attention_mask=causal_mask,
                                      position_ids=position_ids,
                                      past_key_values=None,
                                      cache_position=cache_position,
                                      position_embeddings=position_embeddings,)
    
    if output_layer == 0:
        hidden_states = decoder.norm(hidden_states)
    
    return hidden_states


def custom_move(model, 
                torch_device, 
                output_layer: int=0):
    end_l_idx = output_layer if output_layer != 0 else model.config.num_hidden_layers
    
    model.model.embed_tokens.to(torch_device)
    model.model.rotary_emb.to(torch_device)
    
    for l_idx in range(0, end_l_idx):
        model.model.layers[l_idx].to(torch_device)
        
    if output_layer == 0:
        model.model.norm.to(torch_device)

        
def extract_output_feat(feats: torch.Tensor,
                        labels: torch.Tensor,
                        attn_mask: torch.Tensor):
    label_list, feat_list = [], []
    for s_idx in range(feats.shape[0]):
        tmp_mask = attn_mask[s_idx]
        tmp_feat = feats[s_idx, tmp_mask==1, :]
        tmp_label = labels[s_idx, tmp_mask==1]
        
        feat_list.append(tmp_feat)
        label_list.append(tmp_label)
    return feat_list, label_list


def save_feature_batch(save_dir: str,
                       batch_idx: int,
                       feat_list: list[torch.Tensor],
                       label_list: list[torch.Tensor]):
    from safetensors.torch import save_file

    length_list = [0] + [t.shape[0] for t in feat_list]
    offsets = torch.cumsum(torch.tensor(length_list).long(), dim=0)
    
    tensors = {
        'feats': torch.cat(feat_list, dim=0),
        'labels': torch.cat(label_list, dim=0),
        'offsets': offsets,
    }

    save_path = os.path.join(save_dir, f"batch_{batch_idx:06d}.safetensors")
    tmp_path = save_path + ".tmp"
    save_file(tensors, tmp_path, metadata={"num_samples": str(len(feat_list))})
    os.replace(tmp_path, save_path)


def main(args):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"deepseek-ai/{args.base_model}")
    
    # load dataset
    raw_data = load_dataset(f"ise-uiuc/{args.task}", split="train")
    

    if args.filter_length:
        tmp_id_path = os.path.join('auxiliary_data', f'{args.task}_{args.base_model}_filtered_ids.npz')
        filtered_train_ids = load_filtered_ids(tmp_id_path)
        raw_data = raw_data.filter(lambda _, idx: idx not in filtered_train_ids, with_indices=True)

    formatted_trainset = raw_data.map(code_prompt_completion_converter,
                                      batched=True,
                                      remove_columns=list(raw_data.column_names))
    
    tokenized_dataset = formatted_trainset.map(tokenize_fn,
                                               fn_kwargs={"processing_class": tokenizer},
                                               num_proc=16,
                                               remove_columns=list(formatted_trainset.column_names))
    
    my_collate_fn = partial(prompt_completion_collate_fn, 
                            pad_token_id=tokenizer.pad_token_id,
                            pad_to_multiple_of=8)
    train_loader = DataLoader(tokenized_dataset,
                              collate_fn=my_collate_fn,
                              batch_size=args.batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=False,
                              sampler=RandomSampler(tokenized_dataset))
    
    # load model
    model = AutoModelForCausalLM.from_pretrained(f"deepseek-ai/{args.base_model}",
                                                 dtype=torch.bfloat16,
                                                 attn_implementation=args.attn_impl)
    assert args.output_layer <= model.config.num_hidden_layers, (
        f"--output-layer must be <= {model.config.num_hidden_layers}, got {args.output_layer}"
    )
    model.eval()
    custom_move(model, args.device, output_layer=args.output_layer)
    
    # inference
    if args.save_dir is not None:
        for path in Path(args.save_dir).glob('batch_*.safetensors'):
            path.unlink()

    batch_idx = 0

    with torch.inference_mode():
        for batch in tqdm(train_loader):
            batch_input_ids = batch['input_ids'].to(args.device, non_blocking=True)
            batch_attention_mask = batch['attention_mask'].to(args.device, non_blocking=True)
            
            output_feats = custom_forward(model.model,
                                          batch_input_ids, 
                                          batch_attention_mask,
                                          output_layer=args.output_layer)
            output_feats = output_feats.detach().cpu()

            out_feat_list, label_list = extract_output_feat(output_feats,
                                                            batch['labels'],
                                                            batch_attention_mask.cpu())
            if args.save_dir is not None:
                save_feature_batch(args.save_dir, batch_idx, out_feat_list, label_list)
            
            batch_idx += 1


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
    
    # inference relevant
    parser.add_argument('--batch-size', type=int, default=32)
    
    # others
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-root', type=str, default=None,)
    
    args = parser.parse_args()
    
    assert args.output_layer >= 0, f"--output-layer must be non-negative, got {args.output_layer}"
    
    args.device = torch.device('cuda')
    args.suffix = ''

    if args.save_root is None:
        args.save_root = 'checkpoints'
    
    if args.save_root is not None:
        cache_name = 'feature_cache'
        args.save_dir = os.path.join(args.save_root, args.base_model, args.task, 
                                    f'FT-O{args.output_layer}{args.suffix}', 
                                    cache_name)
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = None
    
    logging.info(args)
    
    set_seed(args.seed)
    
    main(args)
