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
from ..profiles import ExportProfile

logger = get_logger("clipper_local")


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -t duration → re-encoded mp4 with audio fades and strict sync."""
    duration = end - start
    
    # Professional Audio Mastering Chain
    # 1. agate: Silence removal (noise gate)
    # 2. afftdn: Noise suppression
    # 3. acompressor: Dynamic range compression
    # 4. loudnorm: EBU R128 Loudness Normalization & Peak Limiter (Target -16 LUFS, True Peak -1.5dB)
    # 5. afade: 150ms J-Cut fades
    
    fade_duration = 0.15
    fade_out_start = max(0, duration - fade_duration)
    
    filters = [
        "agate=range=0.01:threshold=0.02:attack=2:release=150",
        "afftdn=nf=-25",
        "acompressor=threshold=-15dB:ratio=3:attack=5:release=50",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        f"afade=t=in:st=0:d={fade_duration}",
        f"afade=t=out:st={fade_out_start}:d={fade_duration}"
    ]
    audio_filter = ",".join(filters)
    
    logger.debug(f"Cutting subclip from {start} to {end} for {source_path}")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-threads", "0",
        "-vsync", "1",
        "-async", "1",
        "-avoid_negative_ts", "make_zero",
        "-af", audio_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str, profile: Optional[ExportProfile] = None) -> str:
    """Crop the cut clip to the target aspect ratio, using cinematic tracking."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "opencv-python and numpy are required for --mode local. Install with:\n"
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
        logger.info("Using MediaPipe Face Detection for cinematic tracking.")
    except ImportError:
        logger.warning("MediaPipe not installed. Falling back to Haar Cascades.")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    camera_x: float = float(src_w // 2)
    camera_y: float = float(src_h // 2)
    
    # Virtual camera params
    dead_zone_x = crop_w * 0.15
    max_speed = 15.0  # px per frame
    spring_tension = 0.08  # interpolation factor
    
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate scene motion (saliency)
        motion_map = None
        if prev_gray is not None:
            motion_map = cv2.absdiff(prev_gray, gray)
        prev_gray = gray
        
        target_x, target_y = None, None
        faces = []

        if mp_face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_detection.process(rgb_frame)
            if results.detections:
                for d in results.detections:
                    bbox = d.location_data.relative_bounding_box
                    fx = int(bbox.xmin * src_w)
                    fy = int(bbox.ymin * src_h)
                    fw = int(bbox.width * src_w)
                    fh = int(bbox.height * src_h)
                    faces.append((fx, fy, fw, fh))

        if not faces:
            faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (fx, fy, fw, fh) in faces_haar:
                faces.append((fx, fy, fw, fh))

        if faces:
            # We have faces. Let's find the active speaker using motion weighting.
            if motion_map is not None:
                best_face = None
                max_motion = -1
                for (fx, fy, fw, fh) in faces:
                    fx_clamped = max(0, fx)
                    fy_clamped = max(0, fy)
                    fw_clamped = min(src_w - fx_clamped, fw)
                    fh_clamped = min(src_h - fy_clamped, fh)
                    
                    if fw_clamped > 0 and fh_clamped > 0:
                        face_roi = motion_map[fy_clamped:fy_clamped+fh_clamped, fx_clamped:fx_clamped+fw_clamped]
                        motion_score = np.sum(face_roi) / (fw_clamped * fh_clamped)
                        if motion_score > max_motion:
                            max_motion = motion_score
                            best_face = (fx, fy, fw, fh)
                
                if best_face:
                    fx, fy, fw, fh = best_face
                    target_x = fx + fw // 2
                    target_y = fy + fh // 2
            else:
                # No motion map yet, just pick the largest face
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                target_x = fx + fw // 2
                target_y = fy + fh // 2
                
        else:
            # No faces detected. Fallback to scene saliency (motion center).
            if motion_map is not None:
                # Threshold to find areas of significant motion
                _, thresh = cv2.threshold(motion_map, 25, 255, cv2.THRESH_BINARY)
                M = cv2.moments(thresh)
                if M["m00"] > 0:
                    target_x = int(M["m10"] / M["m00"])
                    target_y = int(M["m01"] / M["m00"])

        if target_x is not None and target_y is not None:
            # Apply Dead-Zone
            dx = target_x - camera_x
            
            # If target is outside dead-zone, move camera
            if abs(dx) > dead_zone_x:
                # Determine direction and magnitude outside dead-zone
                sign = 1 if dx > 0 else -1
                dist_to_move = abs(dx) - dead_zone_x
                
                # Spring interpolation
                velocity = dist_to_move * spring_tension
                
                # Clamp max speed
                velocity = min(velocity, max_speed)
                
                camera_x += sign * velocity

        # Clamp camera to video bounds
        camera_x = max(crop_w / 2, min(src_w - crop_w / 2, camera_x))
        camera_y = max(crop_h / 2, min(src_h - crop_h / 2, camera_y))

        # Convert to top-left corner for cropping
        x0 = int(camera_x - crop_w / 2)
        y0 = int(camera_y - crop_h / 2)
        
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()
    if mp_face_detection:
        mp_face_detection.close()

    # Mux audio from the cut clip back onto the silent reframed video, and apply profile encoding
    if profile is None:
        from ..profiles import get_profile
        profile = get_profile("youtube")
        
    logger.debug(f"Reframing complete. Muxing audio for {out_path} using profile codec {profile.codec}")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-threads", "0",
        "-c:v", profile.codec,
        "-b:v", profile.bitrate,
        "-r", str(profile.fps),
        "-s", profile.resolution,
        "-preset", "fast",
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
    profile: Optional[ExportProfile] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path."""
    if os.path.exists(out_path):
        logger.info(f"Skipping cropping, output already exists: {out_path}")
        return out_path
        
    cut_path = out_path + ".cut.mp4"
    try:
        if not os.path.exists(cut_path):
            _cut_subclip(source_path, start_time, end_time, cut_path)
        _reframe_vertical(cut_path, out_path, aspect_ratio, profile)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path



