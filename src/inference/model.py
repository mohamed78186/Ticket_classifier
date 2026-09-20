"""Loading (and introspecting) a local Hugging Face causal LM."""


def load_model(model_name: str | None = None):
    """Load a tokenizer + model pair, defaulting to config.MODEL_NAME."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional local dependencies to load a model") from exc

    from .config import MODEL_NAME

    checkpoint = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    return tokenizer, model


def get_model_device(model):
    """Return the torch device the model's weights currently live on."""
    return model.get_input_embeddings().weight.device
