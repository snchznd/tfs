import math

import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, d: int, d_k: int, d_v: int, causal_mask: bool = True) -> None:
        super().__init__()

        # create matrices
        self.W_q = torch.nn.Parameter(torch.zeros(size=(d, d_k)))
        self.W_k = torch.nn.Parameter(torch.zeros(size=(d, d_k)))
        self.W_v = torch.nn.Parameter(torch.zeros(size=(d, d_v)))

        # initialize matrices
        torch.nn.init.xavier_normal_(self.W_q)
        torch.nn.init.xavier_normal_(self.W_k)
        torch.nn.init.xavier_normal_(self.W_v)

        # create biases
        self.b_q = torch.nn.Parameter(torch.zeros(size=(d_k,)))
        self.b_k = torch.nn.Parameter(torch.zeros(size=(d_k,)))
        self.b_v = torch.nn.Parameter(torch.zeros(size=(d_v,)))

        # create other elements
        self.softmax = torch.nn.Softmax(dim=2)
        self.normalization_factor = math.sqrt(d_k)
        self.causal_mask = causal_mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # compute matrices
        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        # compute un-normalized attention matrix
        A = Q @ K.transpose(1, 2) / self.normalization_factor

        # perform causal masking
        if self.causal_mask:
            seq_len = X.shape[1]
            mask = torch.tril(torch.ones(size=(seq_len,) * 2)) == 0
            A.masked_fill(mask=mask, value=-torch.inf)

        # normalize attention scores
        A = self.softmax(A)

        return A @ V
