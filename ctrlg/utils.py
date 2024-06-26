import torch
from transformers import LogitsProcessor

torch.set_float32_matmul_precision('high')


@torch.compile
def logsumexp(A, dim):
    return torch.logsumexp(A, dim)


@torch.compile
def matmul_log(A, B):
    bd = len(B.shape) - 2
    A_max = torch.amax(A, dim=-1, keepdim=True)
    B_max = torch.amax(B, dim=bd, keepdim=True)
    A = A - A_max
    B = B - B_max
    A.exp_()
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max + B_max)

    return C


@torch.compile
def matmul_loga_b(A, B):
    A_max = torch.amax(A, dim=-1, keepdim=True)
    A = A - A_max
    A.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(A_max)

    return C


@torch.compile
def matmul_a_logb(A, B):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)

    return C


@torch.compile
def distribute_state_weights(E2D, y):
    device = y.device
    _, hidden_states = y.shape
    return y[E2D[:, None],
            torch.arange(0, hidden_states, device=device)[None, :]]


@torch.compile
def aggregate_edge_weights(E2S, y, num_states):
    device = y.device
    _, hidden_states = y.shape
    num_edges = E2S.shape[0]
    E2S_ = E2S[:, None].expand(-1, hidden_states)

    y_out = torch.zeros(num_states, hidden_states, device=device)

    y_out_max = -1e30 * torch.ones(num_states, hidden_states, device=device)
    y_out_max.scatter_reduce_(0, E2S_, y, reduce='amax')
    y_max = y_out_max[E2S[:, None],
        torch.arange(0, hidden_states, device=device)[None, :]]

    y = torch.exp(y - y_max)
    y_out.scatter_reduce_(0, E2S_, y, reduce='sum')
    y_out.log_()
    y_out.nan_to_num(neginf=-1e30)
    y_out += y_out_max

    return y_out


def ends_at(prefix, suffix,
    offset_min, D_cache, dfa_model):
    ans = []
    for s in range(0, len(suffix)):
        offset = len(prefix) - s
        if offset < offset_min:
            break
        state = D_cache[tuple(prefix[:-s])] if s != 0 else D_cache[tuple(prefix)]
        if dfa_model.is_accept(state):
            if s == 0 or tuple(suffix[:s]) == prefix[-s:]:
                ans.append(s)
    return ans


class ConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, hmm_model, dfa_model,
        min_new_tokens, max_new_tokens, prompt_ids, prefix_ids=[], suffix_ids=[],
        temperature=1.0, token_ranges=None, hmm_batch_size=None):

        device = hmm_model.alpha_exp.device
        hidden_states, vocab_size = hmm_model.hidden_states, hmm_model.vocab_size
        num_states = dfa_model.num_states

        neginf = -1e30
        neginf_cuda = neginf * torch.ones(1, device=device)
        alpha_exp, beta, gamma = hmm_model.alpha_exp, hmm_model.beta, hmm_model.gamma
        alpha_exp_t = torch.transpose(alpha_exp, 0, 1)

        if token_ranges is None:
            token_ranges = [[min_new_tokens, max_new_tokens]]
        min_tokens = min_new_tokens
        max_tokens = max([x[1] for x in token_ranges])

        # initialize cache A
        A_cache = {}
        y = gamma.clone()
        for t in range(0, len(prefix_ids)):
            y = y + beta[:, prefix_ids[t]]
            y = matmul_loga_b(y[None, :], alpha_exp).squeeze(0)
        A_cache[tuple(prefix_ids)] = y

        # initialize cache B
        B_cache = torch.empty(len(suffix_ids), hidden_states, device=device)
        y = torch.zeros(hidden_states, device=device)
        for t in range(len(suffix_ids)-1, -1, -1):
            if t != len(suffix_ids) - 1:
                y = matmul_a_logb(alpha_exp, y[:, None]).squeeze(-1)
            y = y + beta[:, suffix_ids[t]]
            B_cache[t, :] = y

        # compute T_weights
        T_mask = dfa_model.T_mask
        VE_mask = dfa_model.VE_mask
        EV_mask = dfa_model.EV_mask
        E2Src, E2Dst = dfa_model.E2Src, dfa_model.E2Dst

        T_weights = matmul_a_logb(T_mask, torch.transpose(beta, 0, 1)) # num_transitions * hidden_states
        T_weights.nan_to_num_(neginf=neginf)

        # initialize cache C
        y_ = torch.full((num_states, hidden_states), neginf, device=device)
        y_[list(dfa_model.accept_states), :] = y
        y = matmul_loga_b(y_, alpha_exp_t) # num_states * hidden_states

        C = torch.empty(max_tokens+1, num_states, hidden_states, device=device)
        C[0, :, :] = y
        for t in range(1, max_tokens+1):
            y = distribute_state_weights(E2Dst, y) # num_transitions * hidden_states
            y = aggregate_edge_weights(E2Src, T_weights + y, num_states=num_states) # num_states * hidden_states
            y = matmul_loga_b(y, alpha_exp_t) # num_states * hidden_states
            C[t, :, :] = y

        # precompute ranges for C_cache
        ranges = set()
        for token_range in token_ranges:
            min_tokens_, max_tokens_ = token_range
            for i in range(min_tokens_, -1, -1):
                ranges.add((i, i + max_tokens_ - min_tokens_))
            for i in range(max_tokens_ - min_tokens_ - 1, -1, -1):
                ranges.add((0, i))
        ranges = list(ranges)
        range_mask = torch.zeros(len(ranges), max_tokens+1, device=device)
        for idx, r in enumerate(ranges):
            range_mask[idx, torch.arange(r[0], r[1]+1)] = 1.0

        C_shape = C.shape
        C = matmul_a_logb(range_mask, torch.flatten(C, start_dim=1, end_dim=2)) # num_ranges * (num_states * hidden_states)
        C = C.view(-1, C_shape[1], C_shape[2])
        C.nan_to_num_(neginf=neginf)

        C_cache = {}
        for idx, r in enumerate(ranges):
            C_cache[r] = C[idx]

        # initialize cache D
        D_cache = {tuple(prefix_ids): dfa_model.initial_state}

        self.A_cache = A_cache
        self.B_cache = B_cache
        self.C_cache = C_cache
        self.D_cache = D_cache

        self.prompt_ids = prompt_ids
        self.prefix_ids = prefix_ids
        self.suffix_ids = suffix_ids

        self.temperature = temperature
        self.token_ranges = token_ranges
        self.hmm_batch_size = hmm_batch_size

        self.dfa_model = dfa_model
        self.hmm_model = hmm_model


    def __call__(self, input_ids, scores):
        input_ids = input_ids[:,len(self.prompt_ids):].tolist()
        prefixes = [tuple(self.prefix_ids + x) for x in input_ids]

        if len(prefixes[0]) > 0:
            selected_idx = [i for i, prefix in enumerate(prefixes)
                if prefix[-1] != self.hmm_model.eos_token_id]
        else:
            selected_idx = [i for i, _ in enumerate(prefixes)]

        logits = torch.log_softmax(scores, dim=-1)

        if len(selected_idx) > 0:
            selected_prefixes = [prefixes[i] for i in selected_idx]
            if len(self.token_ranges) == 1:
                selected_token_ranges = [self.token_ranges[0] for _ in selected_idx]
            else:
                selected_token_ranges = [self.token_ranges[i] for i in selected_idx]

            hmm_batch_size = len(selected_idx) if self.hmm_batch_size is None else min(len(selected_idx), self.hmm_batch_size)
            hmm_logits, hmm_logits_ = self.compute_logits(selected_prefixes, selected_token_ranges, hmm_batch_size)
            hmm_logits -= hmm_logits_

            # ban special tokens that are not in the HMM
            if hmm_logits.shape[1] < logits.shape[1]:
                neginf = torch.full((hmm_logits.shape[0], logits.shape[1]-hmm_logits.shape[1]), -1e30, device=hmm_logits.device)
                hmm_logits = torch.cat((hmm_logits, neginf), dim=1)
            logits[selected_idx, :] += hmm_logits
            logits = torch.log_softmax(logits, dim=-1)

        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        return logits


    # compute logits for next_token
    def compute_logits(self, prefixes, token_ranges, batch_size):

        device = self.hmm_model.alpha_exp.device
        neginf = -1e30
        neginf_cuda = neginf * torch.ones(1, device=device)

        suffix = self.suffix_ids
        generation_offset = len(self.prefix_ids)
        prefix_num, prefix_len = len(prefixes), len(prefixes[0])

        VE_mask, EV_mask, T_mask = self.dfa_model.VE_mask, self.dfa_model.EV_mask, self.dfa_model.T_mask
        A_cache, B_cache, C_cache, D_cache = self.A_cache, self.B_cache, self.C_cache, self.D_cache
        alpha_exp, beta, gamma = self.hmm_model.alpha_exp, self.hmm_model.beta, self.hmm_model.gamma
        hidden_states, vocab_size = self.hmm_model.hidden_states, self.hmm_model.vocab_size

        # update prefix hidden states
        if prefix_len > generation_offset:
            # update A_cache
            A = torch.stack([A_cache[prefix[:-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
            log_probs = torch.stack([beta[:, prefix[-1]] for prefix in prefixes], dim=0) # len(prefixes) * hidden_states
            A += log_probs
            A = matmul_loga_b(A, alpha_exp)
            for i, prefix in enumerate(prefixes):
                A_cache[prefix] = A[i]

            # update D_cache
            for prefix in prefixes:
                next_state = self.dfa_model.next_state(D_cache[prefix[:-1]], prefix[-1])
                D_cache[prefix] = next_state
        else:
            A = torch.stack([A_cache[prefix] for prefix in prefixes], dim=0) # prefix_num * hidden_states

        logits = torch.full((prefix_num, vocab_size), neginf, device=device)

        # gather the list of indices that has at least one more token left before suffix
        generated_tokens = prefix_len - generation_offset
        selected_idx = [prefix_idx for prefix_idx, prefix in enumerate(prefixes)
            if token_ranges[prefix_idx][1] - generated_tokens > 0]
        selected_num = len(selected_idx)
        if len(selected_idx) > 0:
            for batch_idx in range(0, selected_num, batch_size):
                batch_size_ = min(batch_size, selected_num - batch_idx)
                selected_batch = selected_idx[batch_idx: batch_idx+batch_size_]

                A_batch = A[selected_batch] # batch_size_ * hidden_states

                prefixes_batch = [prefixes[i] for i in selected_batch]

                C_batch = []
                for prefix_idx in selected_batch:
                    min_tokens, max_tokens = token_ranges[prefix_idx]
                    remaining_tokens_max = max_tokens - generated_tokens
                    remaining_tokens_min = max(1, min_tokens - generated_tokens)
                    C_batch.append(C_cache[(remaining_tokens_min-1, remaining_tokens_max-1)])
                C_batch = torch.stack(C_batch, dim=0) # batch_size_ * num_states * hidden_states

                C = A_batch[:, None, :] + C_batch # batch_size_ * num_states * hidden_states

                C_shape = C.shape
                C = matmul_log(torch.flatten(C, start_dim=0, end_dim=1), beta) # (batch_size_ * num_states) * vocab_size
                C = C.view(C_shape[0], C_shape[1], -1) # batch_size_ * num_states * vocab_size

                mask = torch.stack([VE_mask[D_cache[prefix]] for prefix in prefixes_batch], dim=0) # prefix_mask, batch_size_ * num_transitions
                mask = mask[:, :, None] * EV_mask[None, :, :] # batch_size_ * num_transitions * num_states
                mask = torch.transpose(mask, 1, 2) # batch_size_ * num_states * num_transitions

                mask_shape = mask.shape
                mask = torch.matmul(torch.flatten(mask, start_dim=0, end_dim=1), T_mask) # (batch_size_ * num_states) * vocab_size
                mask = mask.view(mask_shape[0], mask_shape[1], -1) # batch_size_ * num_states * vocab_size
                mask = torch.nan_to_num(torch.log(mask), neginf=neginf)

                logits_batch = logsumexp(C + mask, dim=1) # batch_size_ * vocab_size

                logits[selected_batch, :] = logits_batch

        # if current prefix already ends with part/none of the suffix;
        for prefix_idx, prefix in enumerate(prefixes):
            min_tokens, max_tokens = token_ranges[prefix_idx]
            offset_min = min_tokens + generation_offset
            offset_max = max_tokens + generation_offset
            offsets = ends_at(prefix, suffix,
                offset_min, D_cache, self.dfa_model)
            for offset in offsets:
                log_prob = logsumexp(A[prefix_idx] + B_cache[offset], dim=0)
                logits[prefix_idx, suffix[offset]] = torch.logaddexp(logits[prefix_idx, suffix[offset]], log_prob)

        # compute normalizing constant; no hmm mini-batch here
        logits_ = matmul_log(A, beta)

        return logits, logits_


def extract_generated_ids(outputs, prompt_ids, suffix_ids, eos_token_id):
    processed_outputs = []

    suffix_ids = tuple(suffix_ids)
    while len(suffix_ids) > 0 and suffix_ids[-1] == eos_token_id:
        suffix_ids = suffix_ids[:-1]
    prompt_ids = tuple(prompt_ids)

    for output_ids in outputs:
        output_ids = tuple(output_ids)
        while output_ids[-1] == eos_token_id:
            output_ids = output_ids[:-1]
        output_ids = output_ids[len(prompt_ids):]

        l = 0
        for k in range(1, min(len(output_ids), len(suffix_ids))+1):
            if output_ids[-k:] == suffix_ids[:k]:
                l = k
        end = None if l == 0 else -l

        output_ids = output_ids[:end]

        processed_outputs.append(output_ids)

    return processed_outputs


# suffix_logits_only: only use logits from the suffix_ids
# suffix_length_cap: set suffix_ids to suffix_ids[:suffix_length_cap] for ranking
# length_penalty: 0.0 --> rank by log-likelihood, 1.0 --> rank by perplexity
def rank_generated_ids(base_model, generated_ids, prompt_ids, suffix_ids,
    suffix_logits_only=False, suffix_length_cap=None, length_penalty=1.0):
    device = base_model.device
    suffix_ids = suffix_ids[:suffix_length_cap]

    # preprocessing input_ids
    input_ids, logits_mask = [], []
    for generated in generated_ids:
        input_ids.append(list(prompt_ids) + list(generated) + list(suffix_ids))
        if suffix_logits_only:
            logits_mask.append([0.0] * len(prompt_ids) + [0.0] * len(generated) + [1.0] * len(suffix_ids))
        else:
            logits_mask.append([0.0] * len(prompt_ids) + [1.0] * len(generated) + [1.0] * len(suffix_ids))

    max_len = max([len(x) for x in input_ids])
    input_ids = [x + [0] * (max_len - len(x)) for x in input_ids]
    input_ids = torch.tensor(input_ids, device=device)
    logits_mask = [x + [0.0] * (max_len - len(x)) for x in logits_mask]
    logits_mask = torch.tensor(logits_mask, device=device)

    # llm forward
    n, d = input_ids.shape
    with torch.no_grad():
        logits = base_model(input_ids).logits[:, :-1, :]
        logits = torch.log_softmax(logits, dim=-1)
        log_probs = logits[
            torch.arange(n)[:, None],
            torch.arange(d-1)[None, :],
            input_ids[:, 1:]]

    norms = torch.sum(logits_mask[:, 1:], dim=-1) ** length_penalty
    log_probs = torch.sum(log_probs * logits_mask[:, 1:], dim=-1) / norms

    generated_ids_sorted = [a for a, b in 
        sorted([(x, y) for x,y in zip(generated_ids, log_probs.tolist())], key=lambda x: x[1], reverse=True)]

    return generated_ids_sorted