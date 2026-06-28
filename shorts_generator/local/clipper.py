"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. For 9:16 we slide a vertical
     window horizontally across the frame to keep faces centred (Haar
     cascade — same approach as the original repo, no external models).
"""
import os
import subprocess
from typing import List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR
from ..models import Highlight
from ..logger import get_logger

logger = get_logger("clipper_local")


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -t duration → re-encoded mp4 with audio."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-avoid_negative_ts", "make_zero",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    mp_face_detection = None
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        logger.info("Using MediaPipe Face Detection for tracking.")
    except ImportError:
        logger.warning("MediaPipe not installed. Falling back to Haar Cascades. Install with `pip install mediapipe`.")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[float, float]] = None
    smoothing = 0.05  # fluid low-pass filter for face tracking
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cx, cy = None, None

        if mp_face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_detection.process(rgb_frame)
            if results.detections:
                largest_face = max(
                    results.detections, 
                    key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height
                )
                bbox = largest_face.location_data.relative_bounding_box
                x = int(bbox.xmin * src_w)
                y = int(bbox.ymin * src_h)
                w = int(bbox.width * src_w)
                h = int(bbox.height * src_h)
                cx = x + w // 2
                cy = y + h // 2

        if cx is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                cx = x + w // 2
                cy = y + h // 2

        if cx is not None and cy is not None:
            if last_center is None:
                last_center = (float(cx), float(cy))
            else:
                lx, ly = last_center
                new_x = lx + (cx - lx) * smoothing
                new_y = ly + (cy - ly) * smoothing
                last_center = (new_x, new_y)

        if last_center is None:
            last_center = (float(src_w // 2), float(src_h // 2))

        fcx, fcy = last_center
        # Convert to int just for the crop boundaries
        icx, icy = int(fcx), int(fcy)
        x0 = max(0, min(src_w - crop_w, icx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, icy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()
    if mp_face_detection:
        mp_face_detection.close()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    if os.path.exists(out_path):
        logger.info(f"Skipping cropping, output already exists: {out_path}")
        return out_path
        
    cut_path = out_path + ".cut.mp4"
    try:
        if not os.path.exists(cut_path):
            _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path



