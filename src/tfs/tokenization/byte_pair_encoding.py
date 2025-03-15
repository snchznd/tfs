import string
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def get_next_token(
    text: str, current_char_idx: int, tokens: Set[str]
) -> Tuple[str, Optional[int]]:
    token_start_idx = current_char_idx
    token_end_idx = token_start_idx

    while token_end_idx < len(text):
        current_candidate = text[token_start_idx : token_end_idx + 1]
        if current_candidate in tokens:
            return current_candidate, token_end_idx + 1
        token_end_idx += 1

    raise ValueError(
        f'No token found in string "{text[token_start_idx::]}".'
        f"\n Set of tokens: {tokens}."
    )


def get_tokens(text: str, tokens: Set[str]) -> List[str]:
    next_token_start = 0
    text_tokens = []
    while next_token_start != len(text):
        token, next_token_start = get_next_token(
            text=text, current_char_idx=next_token_start, tokens=tokens
        )
        text_tokens.append(token)
    return text_tokens


def count_token_pairs(tokens_list: List[str]) -> Dict[Tuple[str, str], int]:
    pairs_to_freq = defaultdict(lambda: 0)
    for idx in range(len(tokens_list) - 1):
        tokens_pair = tokens_list[idx], tokens_list[idx + 1]
        pairs_to_freq[tokens_pair] += 1
    return pairs_to_freq


def by_pair_encoding(
    text: str, nbr_tokens: int, tokens: Set[str] = set(string.printable)
) -> set:
    pass
