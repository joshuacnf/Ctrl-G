import torch
import torch.nn as nn
import numpy as np
from queue import Queue


def set2npset(A, n):
    res = np.zeros((n,), dtype=bool)
    for x in A:
        res[x] = 1
    return res


def edges2G(edges, reverse=False):
    G = {}
    for edge in edges:
        u, v, transition = edge
        if reverse:
            u, v = v, u
        if u not in G:
            G[u] = []
        if len(transition) > 0:
            G[u].append(v)
    return G


def edges2states(edges):
    states = set()
    for edge in edges:
        u, v, _ = edge
        states.add(u)
        states.add(v)
    return states


def edges2dict(edges):
    res = {}
    for edge in edges:
        u, v, transition = edge
        res[(u, v)] = transition
    return res


def DFA_remove_unreachable_states(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    G = edges2G(edges)
    vis = set()
    Q = Queue()
    Q.put(initial_state)
    vis.add(initial_state)
    while not Q.empty():
        u = Q.get()
        for v in G[u]:
            if v not in vis:
                vis.add(v)
                Q.put(v)

    edges_ = [edge for edge in edges
        if (edge[0] in vis and edge[1] in vis)]
    accept_states_ = set([state for state in accept_states
        if state in vis])

    return {
        'edges': edges_,
        'initial_state': initial_state,
        'accept_states': accept_states_
    }


# light-weight version for DFA_merge_undistinguishable_states
def DFA_merge_dead_states(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']
    vocab_size = edges[0][2].shape[0]

    G_rev = edges2G(edges, reverse=True)
    vis = set()
    Q = Queue()
    for state in accept_states:
        Q.put(state)
        vis.add(state)
    while not Q.empty():
        u = Q.get()
        if u not in G_rev:
            continue
        for v in G_rev[u]:
            if v not in vis:
                vis.add(v)
                Q.put(v)

    dead_state = -1
    assert dead_state not in vis

    G = {}
    edges_ = []
    edges_.append((dead_state, dead_state, np.ones((vocab_size,), dtype=bool)))
    for edge in edges:
        u, v, transition = edge
        if u not in vis:
            continue
        if v not in vis:
            if u not in G:
                G[u] = np.zeros((vocab_size,), dtype=bool)
            G[u] = G[u] | transition
        else:
            edges_.append(edge)
    for u in G:
        edges_.append((u, dead_state, G[u]))

    return {
        'edges': edges_,
        'initial_state': initial_state,
        'accept_states': accept_states
    }


def DFA_merge_undistinguishable_states(A, device='cpu'):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']
    vocab_size = edges[0][2].shape[0]

    state2idx, num_states = {}, 0
    for edge in edges:
        u, v, _ = edge
        for x in [u, v]:
            if x not in state2idx:
                state2idx[x] = num_states
                num_states += 1

    G = [np.zeros((vocab_size,), dtype=int) for _ in range(0, num_states)]
    for edge in edges:
        u, v, trans = edge
        u, v = state2idx[u], state2idx[v]
        G[u] += v * trans

    def find(fa, x):
        while x != fa[x]:
            fa[x] = fa[fa[x]]
            x = fa[x]
        return x

    def count(fa):
        vis = set()
        for x in range(0, fa.shape[0]):
            vis.add(find(fa, x))
        return len(vis)

    fa = np.zeros((num_states,), dtype=int)
    root_accept, root_non_accept = None, None
    for state, u in state2idx.items():
        if state in accept_states:
            if root_accept is None:
                root_accept = u
            fa[u] = root_accept
        else:
            if root_non_accept is None:
                root_non_accept = u
            fa[u] = root_non_accept
    fa_num = count(fa)
    fa_G = fa[G]

    while True:
        partitions = {}
        for u in range(0, num_states):
            fu = find(fa, u)
            if fu not in partitions:
                partitions[fu] = []
            partitions[fu].append(u)

        fa_new = np.arange(0, num_states, dtype=int)
        for _, partition in partitions.items():
            for i, u in enumerate(partition):
                for j in range(i+1, len(partition)):
                    v = partition[j]
                    if np.array_equal(fa_G[u], fa_G[v]):
                        fu, fv = find(fa_new, u), find(fa_new, v)
                        fa_new[fu] = fv

        fa_new_num = count(fa_new)
        if fa_new_num == fa_num:
            break
        fa = fa_new
        fa_num = fa_new_num
        fa_G = fa[G]

    edge2transition = {}
    for edge in edges:
        u, v, transition = edge
        u, v = find(fa, state2idx[u]), find(fa, state2idx[v])
        if (u, v) not in edge2transition:
            edge2transition[(u, v)] = np.zeros((vocab_size,), dtype=bool)
        edge2transition[(u, v)] |= transition

    edges_ = [(k[0], k[1], v) for k, v in edge2transition.items()]
    initial_state_ = find(fa, state2idx[initial_state])
    accept_states_ = set(find(fa, state2idx[state]) for state in accept_states)

    return {
        'edges': edges_,
        'initial_state': initial_state_,
        'accept_states': accept_states_,
    }


def DFA_minimize(A):
    A = DFA_remove_unreachable_states(A)
    A = DFA_merge_dead_states(A)
    A = DFA_merge_undistinguishable_states(A)
    return A


def DFA_size(A):
    edge_cnt = len(A['edges'])
    states = set()
    for edge in A['edges']:
        states.add(edge[0])
        states.add(edge[1])
    state_cnt = len(states)

    return state_cnt, edge_cnt


def DFA_negate(A):
    edges = A['edges']
    initial_state = A['initial_state']
    accept_states = A['accept_states']

    all_states = set()
    for edge in edges:
        u, v, _ = edge
        all_states.add(u)
        all_states.add(v)

    accept_states_ = all_states.difference(accept_states)

    return {
        'edges': edges,
        'initial_state': initial_state,
        'accept_states': accept_states_
    }


def _rename_states(A, f):
    def apply(x, f):
        return f[x] if x in f else x

    edges_ = [(
        apply(edge[0], f),
        apply(edge[1], f),
        edge[2]
    ) for edge in A['edges']]

    initial_state_ = apply(A['initial_state'], f)

    accept_states_ = set([apply(state, f) for state in A['accept_states']])

    return {
        'edges': edges_,
        'initial_state': initial_state_,
        'accept_states': accept_states_,
    }


def _reindex_states(A, next_idx=0):
    states = edges2states(A['edges'])
    f = {}
    for state in states:
        f[state] = next_idx
        next_idx += 1
    return _rename_states(A, f), next_idx


def _copy_state(A, s, count, next_idx=0):
    new_edges = []
    new_states = [s]
    new_states.extend([next_idx+i for i in range(0, count)])
    for edge in A['edges']:
        if edge[0] == s:
            new_edges.extend([(next_idx+i, edge[1], edge[2]) for i in range(0, count)])

    return {
        'edges': A['edges'] + new_edges,
        'initial_state': A['initial_state'],
        'accept_states': A['accept_states']
    }, new_states, next_idx+count


def DFA_concatenate_binary(A, B):
    A, next_idx = _reindex_states(A, next_idx=0)
    B, next_idx = _reindex_states(B, next_idx=next_idx)

    accept_states_A = list(A['accept_states'])
    initial_state_B = B['initial_state']

    A['edges'] = [edge for edge in A['edges'] if edge[0] not in accept_states_A]
    B, new_states, _ = _copy_state(B, B['initial_state'], len(accept_states_A)-1, next_idx=next_idx)
    A = _rename_states(A, {x:y for x,y in zip(accept_states_A, new_states)})

    edges_AB = A['edges'] + B['edges']

    return {
        'edges': edges_AB,
        'initial_state': A['initial_state'],
        'accept_states': B['accept_states'],
    }


def DFA_concatenate(dfa_graphs):
    if dfa_graphs == []:
        return []
    if len(dfa_graphs) == 1:
        return dfa_graphs[0]
    return DFA_concatenate_binary(dfa_graphs[0], DFA_concatenate(dfa_graphs[1:]))


def DFA_prod_binary(A, B, mode='intersection'):
    states_A = edges2states(A['edges'])
    states_B = edges2states(B['edges'])
    states_AB = [(ua, ub) for ua in states_A for ub in states_B]

    EA = edges2dict(A['edges'])
    EB = edges2dict(B['edges'])
    edges_AB = []
    for u in states_AB:
        for v in states_AB:
            ua, ub = u
            va, vb = v
            if (ua, va) in EA and (ub, vb) in EB:
                transition = EA[(ua, va)] & EB[(ub, vb)]
                if transition.any():
                    edges_AB.append((u, v, transition))

    assert mode in ['intersection', 'union']

    initial_state_AB = (A['initial_state'], B['initial_state'])
    if mode == 'intersection':
        accept_states_AB = set([u for u in states_AB
            if u[0] in A['accept_states'] and u[1] in B['accept_states']])
    if mode == 'union':
        accept_states_AB = set([u for u in states_AB
            if u[0] in A['accept_states'] or u[1] in B['accept_states']])

    dfa_graph = DFA_minimize({
        'edges': edges_AB,
        'initial_state': initial_state_AB,
        'accept_states': accept_states_AB,
    })

    return dfa_graph


def DFA_prod(dfa_graphs, mode='intersection'):
    if dfa_graphs == []:
        return []
    if len(dfa_graphs) == 1:
        return dfa_graphs[0]
    return DFA_prod_binary(dfa_graphs[0], DFA_prod(dfa_graphs[1:], mode=mode), mode=mode)


class KMPBuilder:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


    def build(self, pat):

        def compute_lps_i(pattern, lps, l, x):
            if x == pattern[l]:
                l += 1
            else:
                while l != 0:
                    l = lps[l - 1]
                    if x == pattern[l]:
                        l += 1
                        break
            return l

        def compute_lps(pattern):
            m = len(pattern)
            lps = [0] * m
            l = 0
            for i in range(1, m):
                l = compute_lps_i(pattern, lps, l, pattern[i])
                lps[i] = l
            return tuple(lps)

        lps = compute_lps(pat)

        pat_tokens_set = set(pat)
        candidate_tokens = set2npset(pat_tokens_set, self.vocab_size)

        E = {}
        for u in range(0, len(pat)):
            for token in pat_tokens_set:
                if token == pat[u]:
                    v = u + 1
                else:
                    v = 0 if u == 0 else compute_lps_i(pat, lps, lps[u-1], token)

                if (u, v) not in E:
                    E[(u, v)] = np.zeros((self.vocab_size,), dtype=bool)
                E[(u, v)][token] = 1

            if (u, 0) not in E:
                E[(u, 0)] = np.zeros((self.vocab_size,), dtype=bool)
            E[(u, 0)] |= ~candidate_tokens

        E[(len(pat), len(pat))] = np.ones((self.vocab_size,), dtype=bool)

        edges = []
        for e, transition in E.items():
            if transition.any():
                u, v = e
                edges.append((u, v, transition))

        initial_state = 0
        accept_states = set([len(pat)])

        return {
            'edges': edges,
            'initial_state': initial_state,
            'accept_states': accept_states
        }


class AhoCorasickBuilder:
    def __init__(self, vocab_size):
        self.vocab_set = np.ones((vocab_size,), dtype=bool)
        self.vocab_size = vocab_size


    def remove_redundant_patterns(self, patterns):
        vis = set()
        patterns = set(','.join(str(x) for x in pattern) for pattern in patterns)
        patterns = list(patterns)

        for i, a in enumerate(patterns):
            for j in range(i+1, len(patterns)):
                b = patterns[j]
                if a.find(b) != -1:
                    vis.add(a)
                if b.find(a) != -1:
                    vis.add(b)

        return [[int(x) for x in pattern.split(',')]
            for pattern in patterns if pattern not in vis]


    def build(self, patterns):
        vocab_size = self.vocab_size

        # WLOG remove unnecessary patterns
        patterns = self.remove_redundant_patterns(patterns)
        patterns_set = set([tuple(x) for x in patterns])

        # first build trie
        T = {}
        candidate_tokens = set()
        for pattern in patterns:
            cur_state = tuple()
            for token in pattern:
                candidate_tokens.add(token)
                if cur_state not in T:
                    T[cur_state] = {}
                T[cur_state][token] = cur_state + (token,)
                cur_state = cur_state + (token,)
            T[cur_state] = {}

        # augment T to be Aho-Corasick automaton
        Q = Queue()
        fail = {tuple():tuple()}
        for _, v in T[tuple()].items():
            Q.put(v)
        while not Q.empty():
            u = Q.get()
            for token in candidate_tokens:
                if token in T[u]:
                    fail_u = fail[u] if u in fail else tuple()
                    fail[T[u][token]] = T[fail_u][token] if token in T[fail_u] else tuple()
                    Q.put(T[u][token])
                else:
                    fail_u = fail[u] if u in fail else tuple()
                    T[u][token] = T[fail_u][token] if token in T[fail_u] else tuple()

        trans = {}
        for u in T:
            if u in patterns_set:
                continue
            other_tokens = np.ones((vocab_size,), dtype=bool)
            for token, v in T[u].items():
                if (u, v) not in trans:
                    trans[(u, v)] = np.zeros((vocab_size,), dtype=bool)
                trans[(u, v)][token] = 1
                other_tokens[token] = 0
            if other_tokens.any():
                if (u, tuple()) not in trans:
                    trans[(u, tuple())] = other_tokens
                else:
                    trans[(u, tuple())] |= other_tokens

        edges = [(k[0], k[1], v) for k, v in trans.items()]
        for pattern in patterns_set:
            edges.append((pattern, pattern, np.ones((vocab_size,), dtype=bool))) # add self-loops for leaf nodes

        return {
            'edges': edges,
            'initial_state': tuple(),
            'accept_states': patterns_set,
        }


# A placeholder DFA that enforce no constraints
class TrivialBuilder:
    def __init__(self, tokenizer, vocab_size,
            eos_token_id=2):

        vocab_set = np.ones((vocab_size,), dtype=bool) # set([x for x in range(0, vocab_size)])

        self.dfa_graph = {
            'edges': [(0, 1, vocab_set),
                        (1, 0, vocab_set)],
            'initial_state': 0,
            'accept_states': set([0, 1]),
        }


    def build(self):
        return self.dfa_graph


# EOS token must be followed by EOS token
class EOSBuilder:
    def __init__(self, vocab_size, eos_token_id):
        vocab_set = np.ones((vocab_size,), dtype=bool)
        eos = set2npset([eos_token_id], vocab_size)
        others = ~eos

        self.dfa_graph = {
            'edges': [(0, 1, eos),
                    (0, 0, others),
                    (1, 1, eos),
                    (1, 2, others),
                    (2, 2, vocab_set)],
            'initial_state': 0,
            'accept_states': set([0, 1]),
        }


    def build(self):
        return self.dfa_graph


# Ad-hoc implementation of a DFA builder that counts the number of words.
# Here each word is defined as some English characters (i.e. isalpha() gives True)
# seperatred by a character from the sep list. If it does not work as you expected,
# implement your custom WordCountBuilder with this implementation as a reference.
class WordCountBuilder:
    def __init__(self, tokenizer, vocab_size, sep=[' ', '\n', ',', '.', ':', ';', '\"', '/']):
        all_special_ids = set(tokenizer.all_special_ids)
        vocab00, vocab01, vocab10, vocab11 = [np.zeros((vocab_size,), dtype=bool) for _ in range(0, 4)]
        for token_id in range(0, vocab_size):
            if token_id in all_special_ids:
                vocab00[token_id] = 1
                continue

            # special handling for the Llama2 tokenizer; should also work with
            # other tokenizers but not thoroughly tested. The logic here should
            # be using tokenizer.decode(token_id) to convert each token_id to text,
            # but the Llama2 tokenizer automatically removes the leading spaces.
            token = tokenizer.decode([tokenizer.all_special_ids[0], token_id])
            token = token[len(tokenizer.decode(tokenizer.all_special_ids[0])):]

            if token[0] in sep:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab11[token_id] = 1
                else:
                    vocab10[token_id] = 1
            else:
                if any([c.isalpha() or c.isdigit() for c in token]):
                    vocab01[token_id] = 1
                else:
                    vocab00[token_id] = 1

        self.vocab0x = vocab00 | vocab01
        self.vocabx0 = vocab00 | vocab10
        self.vocabx1 = vocab01 | vocab11
        self.vocab10 = vocab10
        self.vocab11 = vocab11
        self.vocab_set = np.ones((vocab_size,), dtype=bool)


    def build(self, min_word_count, max_word_count):
        states = []
        states.extend([(k, s) for k in range(0, max_word_count+1) for s in range(0, 2)])
        states.append((max_word_count+1, 0))

        E = {}
        for u in states:
            k, s = u
            if k <= max_word_count:
                if s == 0:
                    E[(u, u)] = self.vocab0x
                    E[(u, (k, 1))] = self.vocab10
                    E[(u, (k+1, 0))] = self.vocab11
                if s == 1:
                    E[(u, u)] = self.vocabx0
                    E[(u, (k+1, 0))] = self.vocabx1
            else:
                E[(u, u)] = self.vocab_set

        edges = []
        for e, transition in E.items():
            u, v = e
            edges.append((u, v, transition))

        initial_state = (0, 1)
        accept_states = [(k, s) for k in range(min_word_count, max_word_count+1) for s in range(0, 2)]

        return {
            'edges': edges,
            'initial_state': initial_state,
            'accept_states': accept_states,
        }


class DFAModel(nn.Module):
    def __init__(self, dfa_graph, vocab_size):
        super().__init__()

        edges = dfa_graph['edges']
        initial_state = dfa_graph['initial_state']
        accept_states = dfa_graph['accept_states']

        state_cnt, edge_cnt = 0, 0
        state2idx, edge2idx = {}, {}

        # pre-process dfa_graph
        for e in edges:
            u, v, _ = e
            for x in [u, v]:
                if x not in state2idx:
                    state2idx[x] = state_cnt
                    state_cnt += 1
            u_idx, v_idx = state2idx[u], state2idx[v]
            if (u_idx, v_idx) not in edge2idx:
                edge2idx[(u_idx, v_idx)] = edge_cnt
                edge_cnt += 1
            else:
                print('ERROR: duplicate edge!')
                exit(1)

        G = {}
        VE_mask = torch.zeros(state_cnt, edge_cnt)
        EV_mask = torch.zeros(edge_cnt, state_cnt)
        T_mask = torch.zeros(edge_cnt, vocab_size)
        E2Src = torch.tensor([0] * edge_cnt)
        E2Dst = torch.tensor([0] * edge_cnt)
        for e in edges:
            u, v, transition = e    # transition should be a bitset of tokens
            u_idx, v_idx = state2idx[u], state2idx[v]
            edge_idx = edge2idx[(u_idx, v_idx)]
            VE_mask[u_idx, edge_idx] = 1.0
            EV_mask[edge_idx, v_idx] = 1.0
            T_mask[edge_idx, torch.from_numpy(transition)] = 1.0
            E2Src[edge_idx] = u_idx
            E2Dst[edge_idx] = v_idx

            if u_idx not in G:
                G[u_idx] = []
            G[u_idx].append((v_idx, transition))


        self.VE_mask = nn.Parameter(VE_mask, requires_grad=False)
        self.EV_mask = nn.Parameter(EV_mask, requires_grad=False)
        self.T_mask = nn.Parameter(T_mask, requires_grad=False)
        self.E2Src = nn.Parameter(E2Src, requires_grad=False)
        self.E2Dst = nn.Parameter(E2Dst, requires_grad=False)

        self.G = G
        self.num_states = state_cnt
        self.initial_state = state2idx[initial_state]
        self.accept_states = set([state2idx[x] for x in accept_states])


    def next_state(self, state, token):
        for e in self.G[state]:
            v, transition_set = e
            if transition_set[token]:
                return v
        print(f'ERROR: no valid transition! {state} {token}')
        exit(1)


    def is_accept(self, state):
        return state in self.accept_states