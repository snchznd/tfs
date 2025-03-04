import math

import torch


class SelfAttention(torch.nn.Module):
    def __init__(self, d: int, d_k: int, d_v: int) -> None:
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.zeros(size=(d, d_k)))
        self.W_k = torch.nn.Parameter(torch.zeros(size=(d, d_k)))
        self.W_v = torch.nn.Parameter(torch.zeros(size=(d, d_v)))
        torch.nn.init.xavier_normal_(self.W_q)
        torch.nn.init.xavier_normal_(self.W_k)
        torch.nn.init.xavier_normal_(self.W_v)
        self.softmax = torch.nn.Softmax(dim=2)
        self.normalization_factor = math.sqrt(d_k)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        A = self.softmax(Q @ K.transpose(1, 2) / self.normalization_factor)
        return A @ V


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d: int, d_k: int, d_v: int, nbr_heads: int) -> None:
        if d_v % nbr_heads != 0:
            raise ValueError(
                "The output dim needs to be a multiple of the number of attention heads."
            )
        super().__init__()
        self.attention_heads = torch.nn.ModuleList(
            [SelfAttention(d, d_k, d_v // nbr_heads) for _ in range(nbr_heads)]
        )
        self.H = torch.nn.Parameter(torch.zeros(size=(d_v, d_v)))
        torch.nn.init.xavier_normal_(self.H)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.concat([A(X) for A in self.attention_heads], dim=2) @ self.H


# TODO:
# - add causal attention masks
