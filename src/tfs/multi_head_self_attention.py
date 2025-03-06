import torch

from tfs.self_attention import SelfAttention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self, d: int, d_k: int, d_v: int, nbr_heads: int, causal_mask: bool = True
    ) -> None:
        if d_v % nbr_heads != 0:
            raise ValueError(
                "The output dim needs to be a multiple of the number of attention heads."
            )
        super().__init__()
        self.attention_heads = torch.nn.ModuleList(
            [
                SelfAttention(d, d_k, d_v // nbr_heads, causal_mask)
                for _ in range(nbr_heads)
            ]
        )
        self.H = torch.nn.Parameter(torch.zeros(size=(d_v, d)))
        torch.nn.init.xavier_normal_(self.H)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.concat([A(X) for A in self.attention_heads], dim=2) @ self.H
