import string
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def get_next_token(
    text: str,
    current_char_idx: int,
    tokens: Set[str],
    max_token_len: int,
) -> Tuple[str, Optional[int]]:
    """
    Find the longest token from the vocabulary that matches the text starting at current_char_idx.

    This function implements a greedy search algorithm that finds the longest token
    from the vocabulary that matches the text starting at the specified index. It limits
    the search to tokens that are no longer than max_token_len.

    Args:
        text: The input text to tokenize.
        current_char_idx: The index in the text where to start looking for a token.
        tokens: The set of valid tokens to consider.
        max_token_len: The maximum length of tokens to consider.

    Returns:
        A tuple containing:
        - The matched token (str)
        - The index in the text immediately after the matched token (int)

    Raises:
        ValueError: If no token is found at the specified position.
    """
    token_start_idx = current_char_idx
    token_end_idx = min(token_start_idx + max_token_len, len(text))

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
    """
    Tokenize the given text using the provided vocabulary of tokens.

    This function converts a string of text into a list of tokens from the
    given vocabulary. It uses a greedy approach, always selecting the longest
    possible token at each position.

    Args:
        text: The text to tokenize.
        tokens: The set of valid tokens to use for tokenization.

    Returns:
        A list of tokens that, when concatenated, reconstitute the original text.

    Raises:
        ValueError: If a part of the text cannot be tokenized with the given vocabulary.
    """
    next_token_start = 0
    text_tokens = []
    max_token_length = max(len(x) for x in tokens)
    while next_token_start != len(text):
        token, next_token_start = get_next_token(
            text=text,
            current_char_idx=next_token_start,
            tokens=tokens,
            max_token_len=max_token_length,
        )
        text_tokens.append(token)
    return text_tokens


def count_token_pairs(tokens_list: List[str]) -> Dict[Tuple[str, str], int]:
    """
    Count occurrences of adjacent token pairs in a tokenized sequence.

    For each pair of adjacent tokens in the input list, this function counts how many
    times that specific pair occurs. This is a fundamental step in the BPE algorithm
    to identify which token pairs should be merged.

    Args:
        tokens_list: A list of tokens to analyze for adjacent pairs.

    Returns:
        A dictionary mapping token pairs (as tuples) to their frequencies.
        For example: {('a', 'b'): 5, ('b', 'c'): 3, ...}
    """
    pairs_to_freq = defaultdict(lambda: 0)
    for idx in range(len(tokens_list) - 1):
        tokens_pair = tokens_list[idx], tokens_list[idx + 1]
        pairs_to_freq[tokens_pair] += 1
    return pairs_to_freq


def order_token_frequencies(
    frequency_dict: Dict[Tuple[str, str], int],
) -> List[Tuple[str, str, int]]:
    """
    Sort token pairs by frequency in descending order.

    This function takes a dictionary mapping token pairs to their frequencies and
    returns a sorted list of (first_token, second_token, frequency) tuples,
    ordered by descending frequency.

    Args:
        frequency_dict: Dictionary mapping token pairs to frequencies.

    Returns:
        A list of tuples (first_token, second_token, frequency) sorted
        by frequency in descending order.
    """
    return sorted(
        [(k_0, k_1, v) for (k_0, k_1), v in frequency_dict.items()], key=lambda x: -x[2]
    )


def get_ordered_token_pairs_frequencies(
    text: str, tokens: Set[str]
) -> List[Tuple[str, str, int]]:
    """
    Get a sorted list of token pair frequencies from the input text.

    This function:
    1. Tokenizes the input text
    2. Counts adjacent token pairs
    3. Returns pairs sorted by frequency

    Args:
        text: The input text to analyze.
        tokens: The set of tokens to use for tokenization.

    Returns:
        A list of tuples (first_token, second_token, frequency) sorted
        by frequency in descending order, or None if the tokenized text
        contains only one token (making pairs impossible).
    """
    text_tokens_list = get_tokens(text, tokens)

    # if there is only one token, there are no pairs so we return None
    if len(text_tokens_list) == 1:
        return None

    token_pairs_frequencies = count_token_pairs(text_tokens_list)
    ordered_token_pairs_frequencies = order_token_frequencies(token_pairs_frequencies)
    return ordered_token_pairs_frequencies


def get_most_frequent_tokens_pair(text: str, tokens: Set[str]) -> Tuple[str, str]:
    """
    Find the most frequently occurring adjacent token pair in the text.

    This function tokenizes the text and returns the pair of tokens that
    occurs most frequently.

    Args:
        text: The input text to analyze.
        tokens: The set of tokens to use for tokenization.

    Returns:
        A tuple (first_token, second_token) representing the most frequent pair,
        or (None, None) if there are no pairs (e.g., text has only one token).
    """
    token_pairs_frequencies = get_ordered_token_pairs_frequencies(text, tokens)

    # if no pairs frequencies were returned, we return None
    if token_pairs_frequencies is None:
        return None, None

    first_token, second_token, _ = token_pairs_frequencies[0]
    return first_token, second_token


def merge_tokens(first_token: str, second_token: str, tokens: Set[str]) -> None:
    """
    Create a new token by merging two existing tokens.

    This function creates a new token by concatenating the two input tokens,
    and adds it to the set of available tokens.

    Args:
        first_token: The first token in the pair to merge.
        second_token: The second token in the pair to merge.
        tokens: The set of current tokens, which will be modified to include the new token.

    Returns:
        None. The function modifies the tokens set in-place.
    """
    new_token = first_token + second_token
    tokens.add(new_token)


def byte_pair_encoding(
    text: str, target_nbr_tokens: int, initial_tokens: Set[str] = set(string.printable)
) -> Set[str]:
    """
    Implement the Byte Pair Encoding algorithm to create a vocabulary.

    This function repeatedly merges the most frequent pair of tokens in the text
    until either:
    1. The target vocabulary size is reached, or
    2. There are no more token pairs to merge (the algorithm naturally terminates)

    Args:
        text: The input text to analyze for creating the BPE vocabulary.
        target_nbr_tokens: The desired size of the final vocabulary.
        initial_tokens: The initial set of tokens to start with (defaults to printable ASCII).

    Returns:
        The final set of tokens (vocabulary), which includes both the initial tokens
        and the new merged tokens created by the BPE algorithm.

    Notes:
        - The algorithm may terminate before reaching target_nbr_tokens if there
          are no more pairs to merge in the text.
        - The final vocabulary size will never exceed target_nbr_tokens.
    """
    tokens = initial_tokens.copy()
    current_nbr_tokens = len(tokens)
    while current_nbr_tokens < target_nbr_tokens:
        first_token, second_token = get_most_frequent_tokens_pair(text, tokens)

        # if no pairs were returned, there are no more pairs to merge in the
        # text and we simply break the loop
        if first_token is None and second_token is None:
            break

        merge_tokens(first_token, second_token, tokens)
        current_nbr_tokens += 1
    return tokens
