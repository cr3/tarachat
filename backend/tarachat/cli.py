"""Interactive console chat for the RAG system."""

import argparse
import json
import logging
import sys

from tarachat.config import get_settings
from tarachat.models import ChatMessage
from tarachat.rag import RAGSystem, _detect_device


def _ask(rag, query: str, history: list[ChatMessage]) -> str:
    """Send one query, print the streamed response, and return the answer text."""
    print("Assistant: ", end="", flush=True)
    response_parts: list[str] = []

    for event in rag.chat(query, history):
        line = event.removeprefix("data: ").strip()
        if line == "[DONE]":
            break
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if data.get("type") == "token":
            token = data["content"]
            print(token, end="", flush=True)
            response_parts.append(token)
        elif data.get("type") == "sources" and data.get("sources"):
            print("\n\nSources:", flush=True)
            for src in data["sources"]:
                page = src.get("page", "?")
                filename = src.get("filename", "unknown")
                print(f"  {filename}#page={page}", flush=True)

    print("\n", flush=True)
    return "".join(response_parts)


def main():
    parser = argparse.ArgumentParser(description="Chat with the TaraChat RAG system")
    parser.add_argument("prompt", nargs="?", help="Question to ask non-interactively")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    print("Loading RAG system...", flush=True)
    rag = RAGSystem.create(settings=settings, device=_detect_device())

    if args.prompt:
        _ask(rag, args.prompt, [])
        return

    print("Ready. Type your question or Ctrl-D to exit.\n", flush=True)

    history: list[ChatMessage] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not query:
            continue

        answer = _ask(rag, query, history)
        history.append(ChatMessage(role="user", content=query))
        history.append(ChatMessage(role="assistant", content=answer))
