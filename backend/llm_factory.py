import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama


def create_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq."
            )

        model = os.getenv(
            "LLM_MODEL",
            "groq:openai/gpt-oss-120b"
        )

        os.environ["GROQ_API_KEY"] = api_key
        return init_chat_model(model)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        "Choose 'groq' or 'ollama'."
    )
