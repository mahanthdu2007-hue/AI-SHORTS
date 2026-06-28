"""Per-clip cropping via MuAPI /autocrop.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting.
"""
import os
from typing import List

from . import muapi
from .downloader import _extract_video_url
from .models import Highlight
from .logger import get_logger

logger = get_logger("clipper_api")


def crop_clip(source_video_url: str, start_time: float, end_time: float, aspect_ratio: str = "9:16") -> str:
    """Submit one autocrop job and return the URL of the rendered short."""
    payload = {
        "video_url": source_video_url,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "aspect_ratio": aspect_ratio,
    }
    logger.info(f"{start_time:.1f}s → {end_time:.1f}s @ {aspect_ratio}")
    result = muapi.run("autocrop", payload, label=f"autocrop({start_time:.0f}-{end_time:.0f})")
    return _extract_video_url(result)




