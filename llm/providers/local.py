"""LLM backend that runs entirely on the custom in-repo inference engine.

Unlike OpenAICompatible this never touches the network - it drives a
Transformers model directly through ``src.inference``. It exists for
learning purposes and is not what powers the Ollama / cloud-API modes
in the Streamlit app.
"""

from ..base import LLM
from ..model import GenerationRequest
from src.inference.generate import generate_text


class LocalLLM(LLM):
    """Adapter around the hand-rolled generation loop in src/inference."""

    def __init__(self, tokenizer, model) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, request: GenerationRequest) -> str:
        prompt_text = self._render_chat_as_plain_text(request)
        return generate_text(
            tokenizer=self._tokenizer,
            model=self._model,
            prompt=prompt_text,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

    @staticmethod
    def _render_chat_as_plain_text(request: GenerationRequest) -> str:
        """Collapse the chat messages into one "role: content" block per line.

        The local engine has no concept of chat roles - it just
        completes plain text - so this is a simple, readable
        flattening rather than anything resembling real chat
        templating.
        """
        lines = [f"{turn.role.value}: {turn.content}" for turn in request.messages]
        return "\n".join(lines)
