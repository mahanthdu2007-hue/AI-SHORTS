import os
from typing import List, Tuple
from .logger import get_logger

logger = get_logger("scenes")

def detect_scenes(video_path: str) -> List[Tuple[float, float]]:
    """
    Detects scene boundaries in a video using PySceneDetect.
    Returns a list of tuples: (scene_start_seconds, scene_end_seconds).
    If PySceneDetect is not installed, returns an empty list.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector
    except ImportError:
        logger.warning("PySceneDetect not installed. Scene-aware editing will be disabled. Install with `pip install scenedetect`.")
        return []
        
    logger.info(f"Running scene detection on {os.path.basename(video_path)}...")
    
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        
        # Detect scenes
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        
        # Convert FrameTimecode objects to seconds
        scenes_seconds = []
        for scene in scene_list:
            start_sec = scene[0].get_seconds()
            end_sec = scene[1].get_seconds()
            scenes_seconds.append((start_sec, end_sec))
            
        logger.info(f"Detected {len(scenes_seconds)} scenes.")
        return scenes_seconds
    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        return []
