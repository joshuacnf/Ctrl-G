import os
import argparse
import random

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from ctrlg import HMM, loglikelihood


def train_hmm(rank, world_size,
    model_path, checkpoint, save_per_step,
    data_path, dataset, total_chunks, batch_size, sample_length,
    em_schedule, log_file, mask_ratio=0.0, pseudocount=0.1):

    device = f'cuda:{rank}'

    hmm_model = HMM.from_pretrained(f'{model_path}/checkpoint-{checkpoint}').to(device)
    hidden_states, vocab_size = hmm_model.hidden_states, hmm_model.vocab_size

    dev_data = torch.load(f'{args.data_path}/{dataset}.dev')[:, :sample_length]
    dev_size = dev_data.shape[0]
    num_per_process = dev_data.shape[0] // world_size + 1
    dev_data = dev_data[rank * num_per_process: min(dev_data.shape[0], (rank+1) * num_per_process)]

    for step_size, _ in em_schedule:
        assert step_size <= total_chunks

    step_offset = checkpoint
    for step_size, step_count in em_schedule:
        for step_idx in range(0, step_count):
            # evaluate ll
            if step_offset == checkpoint:
                dev_ll = loglikelihood(hmm_model, dev_data, batch_size)
                torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)
                if rank == 0:
                    dev_ll = dev_ll.item() / dev_size
                    msg = f'{checkpoint}\t{-1.0}\t{dev_ll}'
                    print(msg)
                    with open(log_file, 'a+') as fout:
                        fout.write(msg + '\n')

            # get train_step for current step
            train_step = torch.cat([torch.load(f'{data_path}/{dataset}.train.{idx % total_chunks}')
                for idx in range(step_offset, step_offset+step_size)], dim=0)

            # get train_data for current process
            num_per_process = train_step.shape[0] // world_size + 1
            train_data = train_step[rank * num_per_process: min(train_step.shape[0], (rank+1) * num_per_process)]

            # compute flows for one em step
            alpha_flow = torch.zeros(hidden_states, hidden_states, device=device)
            beta_flow = torch.zeros(vocab_size, hidden_states, device=device)
            gamma_flow = torch.zeros(hidden_states, device=device)

            for batch_idx in tqdm(range(0, train_data.shape[0], batch_size)):
                batch_size_ = min(batch_size, train_data.shape[0] - batch_idx)
                train_data_batch = train_data[batch_idx: batch_idx + batch_size_].to(device)

                probs = hmm_model.forward(train_data_batch)
                hmm_model.backward(train_data_batch, probs,
                    alpha_flow, beta_flow, gamma_flow)

            alpha_flow.mul_(hmm_model.alpha_exp)
            beta_flow = torch.permute(beta_flow, (1, 0)).contiguous() # hidden_states * vocab_size

            torch.distributed.all_reduce(alpha_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(beta_flow, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(gamma_flow, op=dist.ReduceOp.SUM)

            # flow to params
            alpha_flow += pseudocount / alpha_flow.shape[-1]
            beta_flow += pseudocount / beta_flow.shape[-1]
            gamma_flow += pseudocount / gamma_flow.shape[-1]
            alpha_exp = alpha_flow / torch.sum(alpha_flow, dim=-1, keepdim=True)
            beta = torch.log(beta_flow / torch.sum(beta_flow, dim=-1, keepdim=True))
            gamma = torch.log(gamma_flow / torch.sum(gamma_flow, dim=0, keepdim=True))

            hmm_model.update_params(alpha_exp, beta, gamma)

            # evaluate ll
            train_ll = loglikelihood(hmm_model, train_data[:dev_data.shape[0]], batch_size)
            dev_ll = loglikelihood(hmm_model, dev_data, batch_size)

            torch.distributed.all_reduce(train_ll, op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(dev_ll, op=dist.ReduceOp.SUM)

            if rank == 0:
                train_ll = train_ll.item() / dev_size
                dev_ll = dev_ll.item() / dev_size
                ckpt = step_offset + step_size
                msg = f'{ckpt}\t{train_ll}\t{dev_ll}'
                print(msg)
                with open(log_file, 'a+') as fout:
                    fout.write(msg + '\n')

                if ckpt % save_per_step == 0 and ckpt != 0:
                    hmm_model.save_pretrained(f'{model_path}/checkpoint-{ckpt}')

            step_offset += step_size


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--checkpoint', default=50, type=int)
    arg_parser.add_argument('--save_per_step', default=150, type=int)

    arg_parser.add_argument('--data_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--total_chunks', default=200, type=int)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--sample_length', default=None, type=int)
    arg_parser.add_argument('--em_schedule', type=str)

    arg_parser.add_argument('--log_file', default='', type=str)

    args = arg_parser.parse_args()

    dist.init_process_group('nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    em_schedule = [tuple([int(y) for y in x.split(',')]) for x in args.em_schedule.split(';') if x != '']

    train_hmm(rank, world_size,
        args.model_path, args.checkpoint, args.save_per_step,
        args.data_path, args.dataset, args.total_chunks, args.batch_size, args.sample_length,
        em_schedule, args.log_file)
