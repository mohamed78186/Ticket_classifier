"""Logit-shaping building blocks used by the generation loop in generate.py.

Each function takes a batch of logits for the *next* token and
returns a modified batch - they're meant to be chained together
(bias, then temperature, then top-k, then top-p) before a token is
finally sampled.
"""

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature; 0 is treated as "leave unchanged" (greedy handles it)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Keep only the top-k logits per row, masking the rest to -inf."""
    if top_k is None:
        return logits
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    vocab_size = logits.shape[-1]
    k = min(top_k, vocab_size)
    kth_largest_value = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_largest_value, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose cumulative
    probability covers ``top_p``, masking everything outside that set."""
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probability = torch.cumsum(sorted_probabilities, dim=-1)

    drop_sorted = cumulative_probability > top_p
    # Always keep at least the single most likely token, even if its own
    # probability already exceeds top_p.
    drop_sorted[..., 1:] = drop_sorted[..., :-1].clone()
    drop_sorted[..., 0] = False

    drop_original_order = torch.zeros_like(drop_sorted).scatter(-1, sorted_indices, drop_sorted)
    return logits.masked_fill(drop_original_order, float("-inf"))


def apply_logit_bias(logits: torch.Tensor, token_bias: dict[int, float]) -> torch.Tensor:
    """Add a fixed bias to specific vocabulary entries before sampling."""
    biased = logits.clone()
    vocab_size = biased.shape[-1]
    for token_id, bias in token_bias.items():
        if not 0 <= token_id < vocab_size:
            raise ValueError(f"token_id out of range: {token_id}")
        biased[..., token_id] += bias
    return biased


def select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Pick the next token: greedy argmax at temperature 0, sampled otherwise."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)
