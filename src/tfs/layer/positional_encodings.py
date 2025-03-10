import torch

from tfs.positional_encodings import get_positional_encodings


class SinusoidalPositionEncodings(torch.nn.Module):
    def __init__(self, max_seq_len: int = None, embedding_dim: int = None) -> None:
        super().__init__()
        self.pos_encodings_pre_computed = False
        if max_seq_len is not None and embedding_dim is not None:
            self.pos_encodings_pre_computed = True
            self.max_seq_len = max_seq_len
            self.embedding_dim = embedding_dim
            pos_encodings = get_positional_encodings(
                n=max_seq_len, d=embedding_dim
            ).unsqueeze(dim=0)
            self.register_buffer("pos_encodings", pos_encodings)

    def forward(self, X):
        nbr_tokens = X.shape[1]
        tokens_dim = X.shape[2]

        positional_encodings = None

        if self.pos_encodings_pre_computed:
            if nbr_tokens > self.max_seq_len:
                raise ValueError(
                    f"Expected sequence of max length {self.max_seq_len} "
                    f"but received sequence of length {nbr_tokens}."
                )

            if tokens_dim != self.embedding_dim:
                raise ValueError(
                    f"Expected embeddings of dimension {self.embedding_dim} "
                    f"but received embeddings of dimension {tokens_dim}."
                )

            positional_encodings = self.pos_encodings[::, :nbr_tokens, ::]

        else:
            positional_encodings = (
                get_positional_encodings(n=nbr_tokens, d=tokens_dim)
                .unsqueeze(dim=0)
                .to(X.device)
            )

        return X + positional_encodings
