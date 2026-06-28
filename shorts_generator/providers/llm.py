import abc
import json
import time
from typing import Dict, Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

from ..config import (
    LLM_PROVIDER,
    OPENAI_MODEL,
    GEMINI_MODEL,
    NVIDIA_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    require_openai_key,
    require_gemini_key,
    require_nvidia_key,
)
from ..logger import get_logger
from .. import muapi

from ..utils import parse_json_robustly

logger = get_logger("provider")

T = TypeVar("T", bound=BaseModel)

class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""
        pass
        
    def generate_structured(self, prompt: str, schema: Type[T]) -> T:
        """Generate structured output adhering to a Pydantic schema with robust JSON parsing."""
        max_retries = 3
        current_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON that matches this schema: " + json.dumps(schema.model_json_schema())
        last_error = None
        
        for attempt in range(max_retries):
            try:
                raw = self.generate_text(current_prompt)
                return parse_json_robustly(raw, schema)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                logger.warning(f"Failed to parse JSON (attempt {attempt+1}/{max_retries}): {e}")
                current_prompt += f"\n\nThe previous attempt failed with error: {e}. Please fix the JSON syntax or missing required fields. Return ONLY valid JSON."
                time.sleep(1) # Backoff
                
        # If we exhausted retries, do not crash the app, raise a handled exception with meaningful error
        raise ValueError(f"LLM failed to return valid JSON matching the schema after {max_retries} attempts. Last validation error: {last_error}")


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str, base_url: str = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai is required. pip install openai") from e
            
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
            
        self.client = OpenAI(**kwargs)
        
    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError("google-genai is required. pip install google-genai") from e
            
        self.client = genai.Client(api_key=require_gemini_key())
        self.model = GEMINI_MODEL
        
    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        return response.text or ""


class OpenAILikeRestProvider(BaseLLMProvider):
    """A generic REST provider for OpenAI-compatible endpoints that doesn't require the openai package."""
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        
    def generate_text(self, prompt: str) -> str:
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"] or ""


class NvidiaProvider(OpenAILikeRestProvider):
    def __init__(self):
        super().__init__(
            api_key=require_nvidia_key(),
            model=NVIDIA_MODEL,
            base_url="https://integrate.api.nvidia.com/v1"
        )


class OllamaProvider(OpenAILikeRestProvider):
    def __init__(self):
        super().__init__(
            api_key="ollama", # API key is ignored by ollama
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )


class MuAPIProvider(BaseLLMProvider):
    """The remote default provider."""
    def __init__(self):
        self.model = "gpt-5-mini" # default in old code
        
    def generate_text(self, prompt: str) -> str:
        result = muapi.run(
            self.model,
            {"prompt": prompt},
            label=self.model,
            timeout=300,
        )

        outputs = result.get("outputs")
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
            return outputs[0]

        for key in ("output", "text", "response", "result", "content"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if isinstance(v, dict):
                inner = v.get("text") or v.get("content")
                if isinstance(inner, str) and inner.strip():
                    return inner
            if isinstance(v, list) and v and isinstance(v[0], str):
                return v[0]

        raise RuntimeError(f"Could not extract text from MuAPI response: {result}")


def get_provider(mode: str) -> BaseLLMProvider:
    """Factory to get the correct LLM provider."""
    if mode == "api":
        return MuAPIProvider()
        
    provider_name = LLM_PROVIDER
    if provider_name == "openai":
        return OpenAIProvider(api_key=require_openai_key(), model=OPENAI_MODEL)
    elif provider_name == "gemini":
        return GeminiProvider()
    elif provider_name == "nvidia":
        return NvidiaProvider()
    elif provider_name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_name}")
