import os
import json
import argparse

import torch
import numpy
import faiss

from tqdm import tqdm
from ctrlg import HMM

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--sequences_file', default='', type=str)
    arg_parser.add_argument('--embeddings_file', default='', type=str)

    arg_parser.add_argument('--hidden_states', type=int)
    arg_parser.add_argument('--vocab_size', type=int)
    arg_parser.add_argument('--eos_token_id', type=int)
    arg_parser.add_argument('--kmeans_iterations', default=100, type=int)
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)

    arg_parser.add_argument('--output_file', default='', type=str)

    args = arg_parser.parse_args()

    return args


def load_examples(sequences_file, embeddings_file, eos_token_id):

    seqs = torch.load(sequences_file).tolist()
    embeddings = torch.load(embeddings_file).float()
    print(f'seqs num: {len(seqs)}')
    print(f'embeddings shape: {embeddings.shape}')

    assert len(seqs) == embeddings.shape[0]
    assert len(seqs[0]) == embeddings.shape[1]

    suffixes, suffix_embeddings = [], []
    for i, seq in enumerate(tqdm(seqs)):
        for j, token in enumerate(seq):
            if token == eos_token_id:
                break
            suffixes.append(((i, j), token, j == len(seq)-1))
            suffix_embeddings.append(embeddings[i, j])

    suffix_embeddings = torch.stack(suffix_embeddings, dim=0)

    return suffixes, suffix_embeddings


def Kmeans_faiss(vecs, K, max_iterations=1000, nredo=1, verbose=True):
    kmeans = faiss.Kmeans(vecs.shape[1], K,
        niter=max_iterations, nredo=nredo, verbose=verbose,
        max_points_per_centroid=vecs.shape[0] // K, gpu=True)
    kmeans.train(vecs)

    return kmeans


def update_flows(alpha, beta, gamma, suffixes, idx2cluster,
        hidden_states, vocab_size, eos_token_id):
    offset2index = {}
    for idx, suffix in enumerate(suffixes):
        offset2index[suffix[0]] = idx

    for idx in tqdm(range(0, len(suffixes))):
        suffix = suffixes[idx]
        suffix_offset, token, is_end = suffix
        suffix_offset_next = (suffix_offset[0], suffix_offset[1]+1)
        u = idx2cluster[idx]

        v = None
        if suffix_offset_next in offset2index:
            v = idx2cluster[offset2index[suffix_offset_next]]
        else:
            v = hidden_states - 1 # the reserved hidden state for <eos> token

        if not is_end:
            alpha[u, v] += 1.0

        beta[u, token] += 1.0
        if suffix_offset[1] == 0:
            gamma[u] += 1.0

    alpha[hidden_states-1, hidden_states-1] = 1.0
    beta[hidden_states-1, eos_token_id] = 1.0


def write_params(alpha_flow, beta_flow, gamma_flow, pseudocount,
    hidden_states, vocab_size, eos_token_id, output_file):

    alpha_flow += pseudocount / alpha_flow.shape[-1]
    beta_flow += pseudocount / beta_flow.shape[-1]
    gamma_flow += pseudocount / gamma_flow.shape[-1]

    alpha_exp = alpha_flow / torch.sum(alpha_flow, dim=-1, keepdim=True)
    beta = torch.log(beta_flow / torch.sum(beta_flow, dim=-1, keepdim=True))
    gamma = torch.log(gamma_flow / torch.sum(gamma_flow, dim=0, keepdim=True))

    hmm_model = HMM(hidden_states, vocab_size, eos_token_id)
    hmm_model.update_params(alpha_exp, beta, gamma)

    hmm_model.save_pretrained(output_file)


def main():
    args = init()

    hidden_states, vocab_size, eos_token_id = args.hidden_states, args.vocab_size, args.eos_token_id

    print(f'loading embeddings from {args.embeddings_file} ...')
    suffixes, suffix_embeddings = load_examples(args.sequences_file, args.embeddings_file, eos_token_id)

    print(suffix_embeddings.shape)
    selected = [idx for idx, suffix in enumerate(suffixes) if suffix[0][1] != 0]
    vecs = suffix_embeddings[selected, :].numpy()
    suffix_embeddings = suffix_embeddings.numpy()
    # vecs = numpy.unique(suffix_embeddings, axis=0) # this operation is slow

    print(f'training K-means with {hidden_states-1} clusters and {vecs.shape[0]} suffix embeddings ...')
    kmeans = Kmeans_faiss(vecs, hidden_states - 1, max_iterations=args.kmeans_iterations)

    print(f'clustering {len(suffixes)} suffix embeddings into {hidden_states-1} clusters ...')
    _, idx2cluster = kmeans.index.search(suffix_embeddings, 1)
    idx2cluster = numpy.squeeze(idx2cluster).tolist()

    alpha = torch.zeros(hidden_states, hidden_states)
    beta = torch.zeros(hidden_states, vocab_size)
    gamma = torch.zeros(hidden_states)

    print('computing flows ...')
    update_flows(alpha, beta, gamma, suffixes, idx2cluster,
        hidden_states, vocab_size, eos_token_id)

    print(f'storing parameters to {args.output_file} ...')
    write_params(alpha, beta, gamma, args.pseudocount,
        hidden_states, vocab_size, eos_token_id, args.output_file)


if __name__ == '__main__':
    main()