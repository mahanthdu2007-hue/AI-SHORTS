import os
import json
import hashlib
import time
import concurrent.futures
from typing import Optional
from tqdm import tqdm

from .models import Result, Transcript, Highlight
from .providers import get_provider, get_subtitle_renderer
from .scenes import detect_scenes
from .highlights import get_highlights
from .profiles import get_profile
from .analyzer import analyze_quality, print_scorecard
from .logger import get_logger

logger = get_logger("pipeline")


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    profile: str = "youtube",
    out_dir_override: Optional[str] = None,
    review_callback=None,
) -> Result:
    """Run the core ViralForge pipeline to generate Shorts from a source video.
    
    This function orchestrates the entire process:
    1. Downloads the source video (or processes a local file).
    2. Transcribes the audio using Whisper.
    3. Identifies highlights via the LLM provider.
    4. Applies scene-aware boundary snapping and padding.
    5. Optionally waits for human review.
    6. Renders the final clips using FFmpeg (audio mastering, cropping, subtitles).
    
    Args:
        youtube_url: Target URL or local file path.
        num_clips: Maximum number of clips to generate.
        aspect_ratio: Target aspect ratio.
        download_format: Source resolution for download.
        language: Forced language code (optional).
        mode: Processing mode ('api' or 'local').
        profile: Export profile ('youtube', 'tiktok', etc.).
        out_dir_override: Custom output directory.
        review_callback: Callback for human review mode.
        
    Returns:
        A Result object containing the generated Shorts and metadata.
    """
    time_total_start = time.perf_counter()
    mode = (mode or "api").lower()
    
    export_profile = get_profile(profile)
    logger.info(f"Using export profile: {export_profile.name} (Res: {export_profile.resolution}, FPS: {export_profile.fps})")
    
    logger.info(f"Initializing ViralForge pipeline in {mode.upper()} mode")
    
    # 1. Dependency Injection Setup
    provider = get_provider(mode)
    subtitle_renderer = get_subtitle_renderer(mode)
    
    if mode == "local":
        from .local.downloader import download_youtube_local
        from .local.transcriber import transcribe_local
        from .local.clipper import crop_clip_local
        download_fn = download_youtube_local
        transcribe_fn = transcribe_local
        crop_clip_fn = crop_clip_local
        from .config import LOCAL_OUTPUT_DIR
        out_dir = out_dir_override or LOCAL_OUTPUT_DIR
    elif mode == "api":
        from .downloader import download_youtube
        from .transcriber import transcribe
        from .clipper import crop_clip
        download_fn = download_youtube
        transcribe_fn = transcribe
        crop_clip_fn = crop_clip
        out_dir = None
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")

    # 2. Download
    logger.info("Starting Download phase")
    t0 = time.perf_counter()
    source_path = download_fn(youtube_url, fmt=download_format)
    t1 = time.perf_counter()
    logger.info(f"✓ Download complete in {t1 - t0:.2f}s")
    
    # 3. Transcribe (with Caching)
    logger.info("Starting Transcription phase")
    t0 = time.perf_counter()
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    url_hash = hashlib.md5(youtube_url.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"transcript_{url_hash}.json")
    
    if os.path.exists(cache_path):
        logger.info("Loading transcript from cache...")
        with open(cache_path, "r", encoding="utf-8") as f:
            transcript = Transcript(**json.load(f))
    else:
        transcript = transcribe_fn(source_path, language=language)
        if not transcript.segments:
            raise RuntimeError("Whisper produced no segments. The video may have no detectable speech.")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f)
            
    t1 = time.perf_counter()
    logger.info(f"✓ Transcript complete in {t1 - t0:.2f}s")
    
    # 3.5 Scene Detection
    logger.info("Starting Scene Detection phase")
    t0 = time.perf_counter()
    scenes = detect_scenes(source_path)
    t1 = time.perf_counter()
    logger.info(f"✓ Scene Detection complete in {t1 - t0:.2f}s")
    
    # 4. Highlights
    logger.info("Starting Highlight Generation phase")
    t0 = time.perf_counter()
    final_clips = get_highlights(transcript, provider, scenes, num_clips=num_clips)
    if not final_clips:
        raise RuntimeError("Highlight generator returned zero clips.")
    t1 = time.perf_counter()
    logger.info(f"✓ AI ranking complete in {t1 - t0:.2f}s")
    
    if review_callback:
        logger.info("Pausing for Human Review...")
        final_clips = review_callback(final_clips, source_path)
        if not final_clips:
            raise RuntimeError("All clips rejected during human review.")
    
    # 5 & 6. Cropping and Subtitles (Parallel Processing)
    logger.info("Starting Rendering phase (Cropping + Subtitles)")
    t0 = time.perf_counter()
    
    if mode == "local" and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    def _render_short(i, short):
        try:
            if mode == "local":
                out_path = os.path.join(out_dir, f"short_{i:02d}{export_profile.output_suffix}.mp4")
                cropped_url = crop_clip_fn(source_path, float(short.start_time), float(short.end_time), aspect_ratio, out_path, export_profile)
            else:
                cropped_url = crop_clip_fn(source_path, float(short.start_time), float(short.end_time), aspect_ratio)
                
            subbed_url = subtitle_renderer.render(
                cropped_url, 
                transcript.model_dump(), 
                cropped_url, 
                start_time=short.start_time,
                profile=export_profile
            )
            short.clip_url = subbed_url
            
            if mode == "local" and out_dir and short.metadata:
                meta_path = os.path.join(out_dir, f"short_{i:02d}_metadata.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(short.metadata.model_dump(), f, indent=2)
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg rendering failed for Clip {i}: {e}")
            short.error = f"Render error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error rendering Clip {i}: {e}", exc_info=True)
            short.error = str(e)
        return short
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_render_short, i, short) for i, short in enumerate(final_clips, 1)]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(final_clips), desc="Rendering Shorts"):
            pass
            
    t1 = time.perf_counter()
    logger.info(f"✓ Rendering complete in {t1 - t0:.2f}s")
    
    # 7. Quality Analysis (Diagnostics)
    logger.info("Running Automatic Quality Analyzer...")
    for i, short in enumerate(final_clips, 1):
        if not short.error:
            report = analyze_quality(short, transcript, provider)
            short.quality_report = report.model_dump()
            print_scorecard(short, report, i)
    
    time_total_end = time.perf_counter()
    logger.info(f"Pipeline finished successfully in {time_total_end - time_total_start:.2f}s")

    return Result(
        mode=mode,
        source_video_url=source_path,
        transcript=transcript,
        highlights=final_clips,
        shorts=final_clips,
    )

