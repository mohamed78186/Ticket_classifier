"""Lazy bridge to the optional Outlines dependency.

Outlines is only needed for locally-constrained structured generation
(see generator.py) - the core package must be importable without it,
so the import happens inside the function rather than at module load
time.
"""


def create_structured_model(model, tokenizer):
    """Wrap a loaded (model, tokenizer) pair as an Outlines model.

    Raises ``RuntimeError`` with an actionable message if the
    ``outlines`` package isn't installed, instead of a raw
    ``ImportError``.
    """
    try:
        import outlines
    except ImportError as exc:
        raise RuntimeError(
            "Outlines is optional. Install the local structured-generation dependencies first."
        ) from exc

    return outlines.from_transformers(model, tokenizer)
