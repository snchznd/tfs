import torch

from tfs.self_attention import SelfAttention


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self, d: int, d_k: int, d_h: int, nbr_heads: int, causal_mask: bool = True
    ) -> None:
        if d_h % nbr_heads != 0:
            raise ValueError(
                "The output dim needs to be a multiple of the number of attention heads."
            )
        super().__init__()
        self.attention_heads = torch.nn.ModuleList(
            [
                SelfAttention(d, d_k, d_h // nbr_heads, causal_mask)
                for _ in range(nbr_heads)
            ]
        )

        # create output projection matrix
        self.H = torch.nn.Parameter(torch.zeros(size=(d_h, d)))

        # initialize output projection matrix
        torch.nn.init.xavier_normal_(self.H)

        # create output projection bias
        self.b_h = torch.nn.Parameter(torch.zeros(size=(d,)))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (
            torch.concat([H(X) for H in self.attention_heads], dim=2) @ self.H
            + self.b_h
        )
