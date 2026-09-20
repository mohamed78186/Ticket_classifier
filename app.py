"""Command-line entry point: classify one ticket, print the JSON result.

Configuration is read entirely from the environment / .env file (see
llm/config.py), so pointing this at a different provider - including
a local Ollama instance - never requires touching this file.
"""

import argparse
import json

from llm.config import LLMSettings
from llm.providers.openai_compatible import OpenAICompatible
from llm.TicketClassifier import TicketClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a support ticket as structured JSON."
    )
    parser.add_argument("ticket", help="Support ticket text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = LLMSettings()
    classifier = TicketClassifier(OpenAICompatible(settings))

    result = classifier.classify(args.ticket)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
