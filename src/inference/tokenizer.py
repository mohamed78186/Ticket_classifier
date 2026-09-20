"""Thin helper so callers don't have to juggle devices by hand."""


def tokenize(tokenizer, text: str, device):
    """Encode ``text`` and move every resulting tensor onto ``device``."""
    encoded = tokenizer(text, return_tensors="pt")
    return {name: tensor.to(device) for name, tensor in encoded.items()}
