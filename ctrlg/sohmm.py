import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def matmul(A, B):
    return torch.matmul(A, B)


def calc_alpha_flow_2nd_order(pf, pp, cp):
    """
    Calculates the alpha flow for a second-order HMM.

    Expected Input Shapes:
    pf: [batch_size, hidden_states, hidden_states] -> (b, i, j)
    pp: [batch_size, hidden_states, hidden_states] -> (b, i, j)
    cp: [batch_size, hidden_states, hidden_states] -> (b, j, k)
    """

    ll = cp.amax(dim=-1)

    pp_exp = torch.exp(pp - ll.unsqueeze(1))

    cp_exp = torch.exp(cp - ll.unsqueeze(-1))

    ratio = pf / pp_exp
    ratio[pp_exp == 0.0] = 0.0

    # Contract over the batch dimension to get the flow for each state pair (i, j)
    af = torch.einsum('bij, bjk -> ijk', ratio, cp_exp)

    return af


class SOHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_states: int, vocab_size: int, eos_token_id: int):
        super().__init__()

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

        # alpha_exp: P(z_{t+1} | z_{t-1}, z_t) -> [H, H, H]
        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states, hidden_states), dim=2)
        # beta: P(x_t | z_t) -> [H, V] (stored in log space)
        beta = torch.log_softmax(torch.randn(hidden_states, vocab_size), dim=1)
        # gamma: P(z_0, z_1) -> [H, H] (stored in log space)
        gamma = torch.log_softmax(torch.randn(hidden_states, hidden_states).flatten(), dim=0).view(hidden_states, hidden_states)

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.gamma = nn.Parameter(gamma, requires_grad=False)


    def update_params(self, alpha_exp, beta, gamma):
        self.alpha_exp.data = alpha_exp
        self.beta.data = beta
        self.gamma.data = gamma


    # bottom-up circuit pass
    def forward(self, input_ids):
        device = self.alpha_exp.device
        alpha_exp, beta = self.alpha_exp, self.beta
        gamma_exp = torch.softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape) # softmax over both dimensions
        hidden_states = self.hidden_states
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous() # seq_len * hidden_states * batch_size
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        ys = []
        y = torch.zeros((hidden_states, hidden_states, batch_size), device=device) # track state pairs (z_{t-1}, z_t, Batch)

        for t in range(seq_len-1, -1, -1):
            if t != seq_len - 1:
                y_max = y.amax(dim=0, keepdim=True).amax(dim=1, keepdim=True)
                y = torch.exp(y - y_max)
                y = torch.einsum('ijk, jkb -> ijb', alpha_exp, y)
                y = torch.log(y) + y_max

            y += input_probs[t, :, :].unsqueeze(0) # broadcast emissions over the z_{t-1} dimension
            ys.append(y)

        y_max = y.amax(dim=0).amax(dim=0)
        y = torch.exp(y - y_max.unsqueeze(0).unsqueeze(0))

        y = torch.einsum('ij, ijb -> b', gamma_exp, y)
        y = torch.log(y) + y_max

        ys.append(y)

        return ys


    # top-down circuit pass
    def backward(self, input_ids, probs, alpha_flow, beta_flow, gamma_flow):
        device = self.alpha_exp.device
        alpha_exp, beta = self.alpha_exp, self.beta
        gamma_exp = torch.softmax(self.gamma.flatten(), dim=0).view(self.gamma.shape)
        hidden_states, vocab_size = self.hidden_states, self.vocab_size
        batch_size, seq_len = input_ids.shape

        input_ids_ = torch.permute(input_ids, (1, 0)).contiguous()
        input_probs = beta[
            torch.arange(0, hidden_states, device=device)[None, :, None],
            input_ids_[:, None, :]].contiguous()
        input_probs *= (input_ids_ != -1)[:, None, :].expand(-1, hidden_states, -1)

        flows = []

        probs_root = torch.permute(probs[-2], (2, 0, 1)).contiguous()
        pf = gamma_exp.unsqueeze(0) * torch.exp(probs_root - probs[-1][:, None, None])
        flows.append(pf)

        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(0, seq_len-1):
            layer_idx = seq_len - t - 1

            pp = probs[layer_idx] - input_probs[t, :, :].unsqueeze(0)
            cp = probs[layer_idx-1]

            pp_b = torch.permute(pp, (2, 0, 1)).contiguous()
            cp_b = torch.permute(cp, (2, 0, 1)).contiguous()

            alpha_flow.add_(calc_alpha_flow_2nd_order(pf, pp_b, cp_b))

            pp_max = pp_b.amax(dim=1, keepdim=True).amax(dim=2, keepdim=True)
            pp_exp = torch.exp(pp_b - pp_max)

            ratio = pf / pp_exp
            ratio[pp_exp == 0.0] = 0.0

            pf_unscaled = torch.einsum('bij, ijk -> bjk', ratio, alpha_exp)
            pf = pf_unscaled * torch.exp(cp_b - pp_max)

            flows.append(pf)

        flows = torch.stack(flows, dim=0)
        flows_zt = torch.sum(flows, dim=2) # marginalize down to current state for emissions
        input_ids_[input_ids_ == -1] = vocab_size
        input_ids_ = input_ids_[:, :, None].expand(-1, -1, hidden_states).view(seq_len * batch_size, hidden_states)

        beta_flow.scatter_add_(0, input_ids_, flows_zt.view(seq_len * batch_size, hidden_states))

    def loglikelihood(self, input_ids, batch_size):
        device = self.alpha_exp.device
        data_size = input_ids.shape[0]

        ll = torch.tensor([0.0], device=device)
        for batch_idx in range(0, data_size, batch_size):
            batch_size_ = min(batch_size, data_size - batch_idx)
            input_ids_batch = input_ids[batch_idx: batch_idx + batch_size_].to(device)
            probs_ = self.forward(input_ids_batch)
            ll += torch.sum(probs_[-1])

        return ll
