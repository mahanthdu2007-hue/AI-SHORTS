"""Local transcription via faster-whisper.

Reads a local media file and returns the same shape the highlight generator
expects: {duration, segments[start, end, text, words]}.
"""
from typing import Optional

from ..config import LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
from ..models import Transcript, TranscriptSegment, TranscriptWord
from ..logger import get_logger

logger = get_logger("transcribe_local")


def _resolve_device() -> str:
    """Determine the optimal compute device (cuda or cpu)."""
    if LOCAL_WHISPER_DEVICE != "auto":
        return LOCAL_WHISPER_DEVICE
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda"
    except (ImportError, OSError, RuntimeError):
        pass
    return "cpu"


def transcribe_local(media_path: str, language: Optional[str] = None) -> Transcript:
    """Transcribe a local media file using faster-whisper, generating word-level timestamps."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"faster-whisper model={LOCAL_WHISPER_MODEL} device={device}")

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)

    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
    
    if LOCAL_WHISPER_VAD_FILTER:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = LOCAL_WHISPER_VAD_PARAMETERS
    else:
        transcribe_kwargs["vad_filter"] = False

    segments_iter, info = model.transcribe(**transcribe_kwargs)

    segments = []
    for s in segments_iter:
        words = []
        if getattr(s, "words", None):
            for w in s.words:
                words.append(TranscriptWord(
                    start=float(w.start),
                    end=float(w.end),
                    word=w.word.strip()
                ))
        
        segments.append(TranscriptSegment(
            start=float(s.start),
            end=float(s.end),
            text=(s.text or "").strip(),
            words=words if words else None
        ))

    duration = float(getattr(info, "duration", 0.0)) or (segments[-1].end if segments else 0.0)
    logger.info(f"{len(segments)} segments, {duration:.0f}s of audio")
    return Transcript(duration=duration, segments=segments)
