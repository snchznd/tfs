import math

import torch


def get_pos_enco_value(n: int, i: int, d: int, L: int = 10_000) -> float:
    if i % 2:
        return math.cos(n / L ** ((i - 1) / d))
    else:

        return math.sin(n / L ** (i / d))


def get_positional_encodings(n: int, d: int) -> torch.Tensor:
    pos_encod = torch.zeros(size=(n, d))
    for position in range(n):
        for coordinate in range(d):
            pos_encod[position, coordinate] = get_pos_enco_value(
                position, coordinate, d
            )
    return pos_encod
