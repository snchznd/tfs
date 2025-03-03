import torch


class Attention(torch.nn.Module):
    def __init__(self, d, d_k, d_v):
        super().__init__()
        self.W_q = torch.rand(size=(d, d_k))
        self.W_k = torch.rand(size=(d, d_k))
        self.W_v = torch.rand(size=(d, d_v))

    def forward(self, X: torch.tensor):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        return Q @ K.T @ V
