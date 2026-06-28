"""Transcription via MuAPI /openai-whisper.

Sends a hosted media URL to MuAPI's Whisper endpoint and returns the segment
shape expected by the highlight generator: {duration, segments[start,end,text]}.
The API runs verbose_json server-side, so we get per-segment timestamps for free.
"""
import json
from typing import Dict, Optional

from . import muapi
from .models import Transcript, TranscriptSegment, TranscriptWord
from .logger import get_logger

logger = get_logger("transcribe_api")


def _coerce_verbose(raw) -> Dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_verbose_payload(result: Dict) -> Dict:
    for key in ("output", "result", "outputs"):
        v = result.get(key)
        if isinstance(v, dict) and "segments" in v:
            return v
        if isinstance(v, list) and v:
            first = v[0]
            decoded = _coerce_verbose(first)
            if "segments" in decoded:
                return decoded
        if isinstance(v, str):
            decoded = _coerce_verbose(v)
            if "segments" in decoded:
                return decoded

    if "segments" in result:
        return result

    raise RuntimeError(f"Could not find Whisper segments in MuAPI response: {result}")


def transcribe(media_url: str, language: Optional[str] = None) -> Transcript:
    logger.info(f"muapi /openai-whisper on {media_url}")
    payload = {
        "audio_url": media_url,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if language:
        payload["language"] = language

    result = muapi.run("openai-whisper", payload, label="openai-whisper")
    verbose = _extract_verbose_payload(result)

    segments = []
    for s in verbose.get("segments") or []:
        words = []
        for w in s.get("words") or []:
            words.append(TranscriptWord(
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                word=(w.get("word") or "").strip()
            ))
            
        segments.append(TranscriptSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=(s.get("text") or "").strip(),
            words=words if words else None
        ))

    duration = float(verbose.get("duration") or (segments[-1].end if segments else 0.0))
    logger.info(f"{len(segments)} segments, {duration:.0f}s of audio")
    return Transcript(duration=duration, segments=segments)

