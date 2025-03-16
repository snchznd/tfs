import string
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def get_next_token(
    text: str, current_char_idx: int, tokens: Set[str]
) -> Tuple[str, Optional[int]]:
    token_start_idx = current_char_idx
    token_end_idx = len(text)

    while token_end_idx > token_start_idx:
        current_candidate = text[token_start_idx:token_end_idx]
        if current_candidate in tokens:
            return current_candidate, token_end_idx
        token_end_idx -= 1

    raise ValueError(
        f'No token found in string "{text[token_start_idx::]}".'
        f"\nSet of tokens: {tokens}."
        f"\nStart of token index: {token_start_idx}."
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


def order_token_frequencies(
    frequency_dict: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str, int]]:
    return sorted(
        [(k_0, k_1, v) for (k_0, k_1), v in frequency_dict.items()], key=lambda x: -x[2]
    )


def get_ordered_token_pairs_frequencies(
    text: str, tokens: Set[str]
) -> List[Tuple[str, str, int]]:
    text_tokens_list = get_tokens(text, tokens)
    token_pairs_frequencies = count_token_pairs(text_tokens_list)
    ordered_token_pairs_frequencies = order_token_frequencies(token_pairs_frequencies)
    return ordered_token_pairs_frequencies


def get_most_frequent_tokens_pair(text: str, tokens: Set[str]) -> Tuple[str, str]:
    token_pairs_frequencies = get_ordered_token_pairs_frequencies(text, tokens)
    first_token, second_token, _ = token_pairs_frequencies[0]
    return first_token, second_token


def merge_tokens(first_token: str, second_token: str, tokens: Set[str]) -> None:
    # tokens.remove(first_token)
    # tokens.remove(second_token)
    new_token = first_token + second_token
    tokens.add(new_token)


def byte_pair_encoding(
    text: str, nbr_tokens: int, tokens: Set[str] = set(string.printable)
) -> Set[str]:
    tokens = tokens.copy()
    current_nbr_tokens = len(tokens)
    acc = 0
    while current_nbr_tokens < nbr_tokens:
        print(f"iteration {acc}")
        acc += 1
        first_token, second_token = get_most_frequent_tokens_pair(text, tokens)
        print(f"\t---> merging tokens <{first_token}> and <{second_token}>.")
        merge_tokens(first_token, second_token, tokens)
        current_nbr_tokens += 1
    return tokens
