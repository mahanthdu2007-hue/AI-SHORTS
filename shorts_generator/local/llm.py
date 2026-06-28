"""Local LLM backend — OpenAI, Gemini or NVIDIA."""

import os

from ..config import (
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai is required.\n"
            "pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend."""
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required.\n"
            "pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )

    return response.text or ""


def call_nvidia_llm(prompt: str) -> str:
    """NVIDIA NIM backend."""

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package is required.\n"
            "pip install openai"
        ) from e

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY in .env")

    model = os.getenv(
        "NVIDIA_MODEL",
        "qwen/qwen3-next-80b-a3b-instruct",
    )

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured provider."""

    provider = (LLM_PROVIDER or "openai").strip().lower()

    if provider == "openai":
        return call_openai_llm(prompt)

    if provider == "gemini":
        return call_gemini_llm(prompt)

    if provider == "nvidia":
        return call_nvidia_llm(prompt)

    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Use 'openai', 'gemini', or 'nvidia'."
    )