"""
Streamlit app for the Ticket Classifier project.

Lets the user pick, from the UI, whether to run inference through:
  1) Ollama running locally (http://localhost:11434) — no real API key needed.
  2) Any OpenAI-compatible cloud API (Groq, OpenAI, etc.) — needs an API key.

Both modes reuse the exact same code path (llm.providers.openai_compatible.OpenAICompatible),
because Ollama exposes an OpenAI-compatible /v1 endpoint. No new provider class was needed —
only different Base URL / API key / model values.

Run with:
    streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

from llm.config import LLMSettings
from llm.exceptions import LLMError
from llm.gateway import LLMGateway
from llm.providers.openai_compatible import OpenAICompatible
from llm.TicketClassifier import TicketClassifier

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="centered")

DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # Ollama doesn't check the key, but the openai library requires a non-empty value
        "model": "llama3.1",
    },
    "api": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "",
        "model": "openai/gpt-oss-20b",
    },
}


def get_ollama_models(root_url: str) -> list[str]:
    """Tries to fetch the list of models actually pulled on the local Ollama instance."""
    try:
        base = root_url.replace("/v1", "").rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def build_classifier(base_url: str, api_key: str, model: str, timeout: float, max_attempts: int) -> TicketClassifier:
    os.environ["LLM_API_KEY"] = api_key or "not-needed"
    os.environ["LLM_BASE_URL"] = base_url
    os.environ["LLM_MODEL"] = model
    os.environ["LLM_TIMEOUT"] = str(timeout)
    os.environ["LLM_MAX_ATTEMPTS"] = str(max_attempts)
    os.environ["LLM_BASE_DELAY"] = "1.0"

    settings = LLMSettings()
    provider = OpenAICompatible(settings)
    gateway = LLMGateway(
        provider,
        max_attempts=settings.max_attempts,
        base_delay=settings.base_delay,
    )
    return TicketClassifier(gateway)


st.title("🎫 Support Ticket Classifier")
st.caption(
    "Automatically classify a support ticket into category, sentiment, urgency, and a summary — "
    "using either Ollama running on your machine, or any OpenAI-compatible API provider (Groq, OpenAI, etc.)."
)

with st.sidebar:
    st.header("⚙️ Run Settings")

    mode_label = st.radio("Model source", ["Ollama (local)", "API (cloud)"], index=0)
    mode_key = "ollama" if mode_label.startswith("Ollama") else "api"
    d = DEFAULTS[mode_key]

    base_url = st.text_input("Base URL", value=d["base_url"])

    if mode_key == "ollama":
        st.caption(
            "Make sure Ollama is running (`ollama serve`) and the model has been pulled "
            "(`ollama pull <model>`) before you hit Classify."
        )
        available_models = get_ollama_models(base_url)
        if available_models:
            model = st.selectbox("Model", available_models)
        else:
            model = st.text_input("Model", value=d["model"])
            st.info("Couldn't fetch the model list from Ollama right now (it may not be running). This will be checked again when you classify.")
        api_key = d["api_key"]
    else:
        model = st.text_input("Model", value=d["model"])
        api_key = st.text_input("API Key", value=d["api_key"], type="password")

    timeout = st.number_input("Timeout (seconds)", value=30, min_value=1)
    max_attempts = st.number_input("Retry attempts on failure", value=3, min_value=1)

ticket_text = st.text_area(
    "Support ticket text",
    height=160,
    placeholder="Example: I was charged twice for the same order and need a refund immediately.",
)

col1, col2 = st.columns(2)
classify_clicked = col1.button("🚀 Classify Ticket", use_container_width=True, type="primary")
test_clicked = col2.button("🔌 Test Connection", use_container_width=True)

if test_clicked:
    with st.spinner("Testing the connection..."):
        try:
            classifier = build_classifier(base_url, api_key, model, timeout, max_attempts)
            classifier.classify("Test connection ticket, please ignore.")
            st.success("Connection is working ✅")
        except LLMError as exc:
            st.error(f"Failed to connect to the model provider: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")

if classify_clicked:
    if not ticket_text.strip():
        st.warning("Please enter the ticket text first.")
    elif mode_key == "api" and not api_key:
        st.warning("You need to enter an API Key to use API mode.")
    else:
        with st.spinner("Classifying the ticket..."):
            try:
                classifier = build_classifier(base_url, api_key, model, timeout, max_attempts)
                result = classifier.classify(ticket_text)
                st.success("Classified successfully ✅")

                urgency_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                sentiment_emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}

                c1, c2, c3 = st.columns(3)
                c1.metric("Category", result.category)
                c2.metric("Sentiment", f"{sentiment_emoji.get(result.sentiment, '')} {result.sentiment}")
                c3.metric("Urgency", f"{urgency_emoji.get(result.urgency, '')} {result.urgency}")

                st.markdown("**Summary:**")
                st.info(result.summary)

                with st.expander("View full JSON"):
                    st.json(result.model_dump())
            except LLMError as exc:
                st.error(f"Model provider error: {exc}")
            except ValueError as exc:
                st.error(f"Could not parse the model's output as valid JSON: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error: {exc}")