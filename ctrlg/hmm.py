import os

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def matmul(A, B):
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    af = torch.matmul(ratio, cp)

    return af


class HMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_states: int, vocab_size: int, eos_token_id: int):
        super().__init__()

        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states), dim=1)
        beta = torch.log_softmax(torch.randn(hidden_states, vocab_size), dim=1)
        gamma = torch.log_softmax(torch.randn(hidden_states), dim=0)

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id


    def update_params(self, alpha_exp, beta, gamma):
        self.alpha_exp.data = alpha_exp
        self.beta.data = beta
        self.gamma.data = gamma


    # bottom-up circuit pass
    def forward(self, input_ids):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1) # 0.0 for MISSING token

        ys = []
        y = torch.zeros((hidden_states, batch_size), device=device)
        for t in range(seq_len-1, -1, -1):
            if t != seq_len - 1:
                y_max = torch.amax(y, dim=0, keepdim=True)
                y = torch.exp(y - y_max)
                y = matmul(alpha_exp, y)
                y = torch.log(y) + y_max
            y += input_probs[t, :, :] # hidden_states * batch_size
            ys.append(y)

        y_max = torch.amax(y, dim=0)
        y = torch.exp(y - y_max.unsqueeze(0))
        y = matmul(gamma_exp.unsqueeze(0), y).squeeze()
        y = torch.log(y) + y_max

        ys.append(y)

        return ys


    # top-down circuit pass
    def backward(self, input_ids, probs,
        alpha_flow, beta_flow, gamma_flow):
        device = self.alpha_exp.device
        alpha_exp, beta, gamma_exp = self.alpha_exp, self.beta, torch.softmax(self.gamma, dim=0)
        hidden_states, vocab_size, eos_token_id = self.hidden_states, self.vocab_size, self.eos_token_id
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous() # seq_len * batch_size
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        flows = []
        pf = gamma_exp.unsqueeze(0) * torch.exp(
            torch.permute(probs[-2], (1, 0)).contiguous() - probs[-1][:, None]) # batch_size * hidden_states
        flows.append(pf)

        # update gamma_flow
        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len-1):
            layer_idx = seq_len - t - 1
            pp = probs[layer_idx] - input_probs[t, :, :] # parent probs; hidden_states * batch_size
            cp = probs[layer_idx-1] # batch_size * hidden_states

            alpha_flow.add_(ib_ib_bj_to_ij(torch.permute(pf, (1, 0)).contiguous(),
                pp,
                torch.permute(cp, (1, 0)).contiguous()))

            pp = torch.permute(pp, (1, 0)) # batch_size * hidden_states
            cp = torch.permute(cp, (1, 0)) # batch_size * hidden_states
            pp_max = torch.amax(pp, dim=1, keepdim=True) # batch_size * 1
            pp_ = torch.exp(pp - pp_max)

            ratio = pf / pp_
            ratio[pp_ == 0.0] = 0.0
            pf = matmul(ratio, alpha_exp) * torch.exp(cp - pp_max)

            flows.append(pf)

        # update beta_flow
        flows = torch.stack(flows, dim=0) # seq_len * batch_size * hidden_states
        input_ids_[input_ids_ == -1] = vocab_size
        input_ids_ = input_ids_[:, :, None].expand(-1, -1, hidden_states).view(seq_len * batch_size, hidden_states)
        beta_flow.scatter_add_(0, input_ids_, flows.view(seq_len * batch_size, hidden_states))


    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size, seq_len = input_ids.shape

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1])

        return ll