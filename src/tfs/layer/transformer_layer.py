import torch

from tfs.attention.multi_head_self_attention import MultiHeadSelfAttention


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d: int,
        d_k: int,
        d_h: int,
        nbr_heads: int,
        causal_mask: bool = True,
        ffn_factor: int = 4,
    ):
        super().__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention(
            d, d_k, d_h, nbr_heads, causal_mask
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=d, out_features=d * ffn_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=d * ffn_factor, out_features=d),
        )

        # torch.nn.Linear(in_features=d, out_features=d)
        self.layer_norm_0 = torch.nn.LayerNorm(normalized_shape=d)
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=d)

    def forward(self, X):
        # apply attention layer and residual connection
        X = self.multi_head_self_attention(X) + X

        # apply first layer normalization
        X = self.layer_norm_0(X)

        # apply MLP and residual connection
        X = self.mlp(X) + X

        # apply second layer normalization
        X = self.layer_norm_1(X)

        return X
