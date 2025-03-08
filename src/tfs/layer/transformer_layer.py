import torch

from tfs.attention.multi_head_self_attention import MultiHeadSelfAttention


class TransformerLayer(torch.nn.Module):
    # should this be renamed to TransfomerSelfAttentionLayer?
    def __init__(
        self,
        d: int,
        d_k: int,
        d_h: int,
        nbr_heads: int,
        causal_mask: bool = True,
        ffn_factor: int = 4,
        p_dropout_attention: float = 0.1,
        p_dropout_mlp: float = 0.1,
    ):
        super().__init__()

        # initialize multi-head self-attention
        self.multi_head_self_attention = MultiHeadSelfAttention(
            d, d_k, d_h, nbr_heads, causal_mask
        )

        # initialize MLP sub-layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=d, out_features=d * ffn_factor),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=d * ffn_factor, out_features=d),
        )

        # initialize layer normalization sub-layers
        self.layer_norm_0 = torch.nn.LayerNorm(normalized_shape=d)
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=d)

        # initialize dropout layers - if specified
        self.dropout_0 = torch.nn.Dropout(p=p_dropout_attention)
        self.dropout_1 = torch.nn.Dropout(p=p_dropout_mlp)

    def forward(self, X):
        # apply attention layer, dropout, and residual connection
        X = self.dropout_0(self.multi_head_self_attention(X)) + X

        # apply first layer normalization
        X = self.layer_norm_0(X)

        # apply MLP, dropout, and residual connection
        X = self.dropout_1(self.mlp(X)) + X

        # apply second layer normalization
        X = self.layer_norm_1(X)

        return X
