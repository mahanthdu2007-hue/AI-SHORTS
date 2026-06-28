"""Utility functions for JSON parsing and generic helpers."""
import json
import re
from typing import Optional, Dict, Any, TypeVar, Type
from urllib.parse import urlparse, parse_qs
from pydantic import BaseModel, ValidationError
from .logger import get_logger

logger = get_logger("utils")
T = TypeVar("T", bound=BaseModel)

def parse_json_robustly(raw: str, schema: Type[T]) -> T:
    """Robustly parse a JSON string into a Pydantic schema, stripping markdown and trailing commas.
    
    Raises:
        json.JSONDecodeError: If the JSON is completely invalid.
        ValidationError: If the JSON is valid but does not match the Pydantic schema.
    """
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    
    start_idx = -1
    if "{" in raw and "[" in raw:
        start_idx = min(raw.find("{"), raw.find("["))
    elif "{" in raw:
        start_idx = raw.find("{")
    elif "[" in raw:
        start_idx = raw.find("[")
        
    end_idx = -1
    if "}" in raw and "]" in raw:
        end_idx = max(raw.rfind("}"), raw.rfind("]"))
    elif "}" in raw:
        end_idx = raw.rfind("}")
    elif "]" in raw:
        end_idx = raw.rfind("]")
        
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        raw = raw[start_idx:end_idx+1]
        
    raw = re.sub(r",(\s*[\]}])", r"\1", raw)
    
    parsed = json.loads(raw)
    return schema(**parsed)


def extract_youtube_video_id(source: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(source)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None
