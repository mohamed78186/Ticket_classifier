"""The hand-written autoregressive generation loop for the local engine.

This exists to show, step by step, what a chat/completions API is
doing under the hood: encode the prompt, repeatedly predict one more
token from the model's logits, and stop on an end token or a
configured stop sequence.
"""

import torch

from .decoder import apply_logit_bias, apply_temperature, apply_top_k, apply_top_p, select_next_token
from .model import get_model_device
from .stopping import StopSequenceDetector
from .tokenizer import tokenize


@torch.no_grad()
def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    token_bias: dict[int, float] | None = None,
) -> str:
    """Generate up to ``max_new_tokens`` tokens of continuation for ``prompt``."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    device = get_model_device(model)
    encoded_prompt = tokenize(tokenizer, prompt, device)
    input_ids = encoded_prompt["input_ids"]
    attention_mask = encoded_prompt["attention_mask"]
    prompt_length = input_ids.shape[1]

    detector = StopSequenceDetector(_encode_stop_sequences(tokenizer, stop_sequences))
    cache = None

    for _ in range(max_new_tokens):
        model_inputs = _next_forward_inputs(input_ids, attention_mask, cache)
        outputs = model(**model_inputs)
        cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        next_token_logits = _shape_logits(next_token_logits, temperature, top_k, top_p, token_bias)
        next_token = select_next_token(next_token_logits, temperature)

        input_ids, attention_mask = _append_token(input_ids, attention_mask, next_token, device)

        if _hit_eos(tokenizer, next_token):
            break
        if _hit_stop_sequence(detector, input_ids, prompt_length):
            break

    new_token_ids = input_ids[:, prompt_length:]
    return tokenizer.decode(new_token_ids[0], skip_special_tokens=True)


def _encode_stop_sequences(tokenizer, stop_sequences: list[str] | None) -> list[list[int]]:
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text in (stop_sequences or [])]
    return [sequence for sequence in encoded if sequence]


def _next_forward_inputs(input_ids, attention_mask, cache) -> dict:
    """Feed the whole prompt on the first step, then just the newest token
    while reusing the KV cache for every step after that."""
    model_inputs = {
        "input_ids": input_ids if cache is None else input_ids[:, -1:],
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    if cache is not None:
        model_inputs["past_key_values"] = cache
    return model_inputs


def _shape_logits(logits, temperature, top_k, top_p, token_bias):
    if token_bias:
        logits = apply_logit_bias(logits, token_bias)
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return logits


def _append_token(input_ids, attention_mask, next_token, device):
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    new_mask_column = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
    attention_mask = torch.cat([attention_mask, new_mask_column], dim=-1)
    return input_ids, attention_mask


def _hit_eos(tokenizer, next_token) -> bool:
    return tokenizer.eos_token_id is not None and torch.all(next_token == tokenizer.eos_token_id).item()


def _hit_stop_sequence(detector: StopSequenceDetector, input_ids, prompt_length: int) -> bool:
    generated_so_far = input_ids[:, prompt_length:]
    return generated_so_far.shape[0] == 1 and bool(detector.match(generated_so_far[0].tolist()))
