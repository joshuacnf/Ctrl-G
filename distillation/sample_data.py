import os
import sys
import json
import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from tqdm import tqdm

import transformers
transformers.utils.logging.get_logger("transformers").setLevel(transformers.utils.logging.ERROR)


def pad_to_len(x, d, eos_token_id):
    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d-x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, eos_token_id, dtype=x.dtype)), dim=1)

    return x


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_name_or_path', default='', type=str)
    arg_parser.add_argument('--tokenizer_name_or_path', default='', type=str)
    arg_parser.add_argument('--cache_prompt_past_key_values', action='store_true')
    arg_parser.add_argument('--half', action='store_true')

    arg_parser.add_argument('--input_file',  default='', type=str)
    arg_parser.add_argument('--chunk_size', default=32, type=int)
    arg_parser.add_argument('--total_chunks',  default=1, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--max_new_tokens', type=int, default=128)
    arg_parser.add_argument('--save_embeddings', action='store_true')

    arg_parser.add_argument('--output_file',  default='', type=str)

    args = arg_parser.parse_args()

    dist.init_process_group('gloo')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f'cuda:{rank}'

    # load base model & tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    if args.half:
        base_model.bfloat16()
    base_model.to(device)
    base_model.eval()

    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path != '' else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, padding_side='left')

    chunk_size_per_process = args.chunk_size // world_size + 1

    # load input_data: a list of prompts for sampling data from the base model
    with open(args.input_file, 'r') as fin:
        input_data = json.load(fin)

    for chunk_idx in range(0, args.total_chunks):
        if rank == 0:
            print(f'generating samples for chunk {chunk_idx} ...')

        sequences, embeddings = [], []
        for prompt_idx, prompt in enumerate(input_data):
            if rank == 0:
                print(f'generating samples for prompt {prompt_idx} ...')

            prompt_ids = tokenizer.encode(prompt)
            chunk_size_per_process_prompt = chunk_size_per_process // len(input_data) + 1
            chunk_size_per_process_prompt = min(chunk_size_per_process_prompt, chunk_size_per_process - prompt_idx * chunk_size_per_process_prompt)

            if args.cache_prompt_past_key_values and len(prompt_ids) > 1:
                with torch.no_grad():
                    past_key_values = base_model(torch.tensor(prompt_ids[:-1], device=device)).past_key_values

            for batch_idx in tqdm(range(0, chunk_size_per_process_prompt, args.batch_size)):
                batch_size_ = min(args.batch_size, chunk_size_per_process_prompt - batch_idx)
                if args.cache_prompt_past_key_values and len(prompt_ids) > 1:
                    model_kwargs = { 'past_key_values': tuple([tuple([col.expand(batch_size_, -1, -1, -1).contiguous()
                        for col in row]) for row in past_key_values]) }
                else:
                    model_kwargs = {}

                with torch.no_grad():
                    generation_output = base_model.generate(
                        input_ids=torch.tensor([prompt_ids] * batch_size_, device=device),
                        do_sample=True, top_k=0, pad_token_id=tokenizer.eos_token_id,
                        max_new_tokens=args.max_new_tokens, output_hidden_states=args.save_embeddings,
                        return_dict_in_generate=True, **model_kwargs
                    )
                    sequences_batch = generation_output.sequences[:,len(prompt_ids):].cpu()
                    sequences_batch = pad_to_len(sequences_batch, args.max_new_tokens, tokenizer.eos_token_id)
                    sequences.append(sequences_batch)

                if args.save_embeddings:
                    last_hidden_states = [hs[-1][:, -1, :] for hs in generation_output.hidden_states] # gather last Layer hidden_states
                    last_hidden_states = torch.stack(last_hidden_states, dim=1).cpu()
                    last_hidden_states = pad_to_len(last_hidden_states, args.max_new_tokens, tokenizer.eos_token_id)
                    embeddings.append(last_hidden_states)

        sequences = torch.cat(sequences, dim=0)
        if rank == 0:
            sequences_list = [torch.empty_like(sequences, dtype=sequences.dtype)
                    for idx in range(world_size)]
            dist.gather(sequences, gather_list=sequences_list)
        else:
            dist.gather(sequences)

        if args.save_embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            if rank == 0:
                embeddings_list = [torch.empty_like(embeddings, dtype=embeddings.dtype)
                    for idx in range(world_size)]
                dist.gather(embeddings, gather_list=embeddings_list)
            else:
                dist.gather(embeddings)

        if rank == 0:
            sequences = torch.cat(sequences_list, dim=0)
            perm = torch.randperm(args.chunk_size)
            sequences = sequences[perm, :]
            output_file = f'{args.output_file}.{chunk_idx}' if args.total_chunks > 1 else f'{args.output_file}'
            torch.save(sequences, output_file)
            if args.save_embeddings:
                embeddings = torch.cat(embeddings_list, dim=0)
                embeddings = embeddings[perm, :]
                torch.save(embeddings, output_file + '.embeddings')