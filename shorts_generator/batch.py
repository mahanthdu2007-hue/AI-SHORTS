import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

from .logger import get_logger
from .pipeline import generate_shorts
from .utils import extract_youtube_video_id

logger = get_logger("batch")

def _is_youtube_playlist(url: str) -> bool:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc and "playlist" in parsed.path:
        return True
    if "youtube.com" in parsed.netloc and "list=" in parsed.query:
        return True
    return False

def _extract_playlist_urls(url: str) -> List[str]:
    logger.info(f"Extracting playlist URLs from {url}...")
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt_dlp not installed. Cannot parse playlist.")
        return [url]
        
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    urls = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    if entry.get('url'):
                        urls.append(entry['url'])
                    elif entry.get('id'):
                        urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
        except Exception as e:
            logger.error(f"Failed to extract playlist: {e}")
            return [url]
            
    logger.info(f"Found {len(urls)} videos in playlist.")
    return urls

def normalize_inputs(inputs: List[str]) -> List[str]:
    """Normalize a mixed list of URLs, files, and folders into a flat list of processable targets.
    
    Args:
        inputs: List of YouTube URLs, playlist URLs, local file paths, or directory paths.
        
    Returns:
        A flattened list of valid targets (URLs or file paths) ready for processing.
    """
    targets = []
    
    for inp in inputs:
        inp = inp.strip()
        
        # Check if local folder
        candidate_path = Path(inp).expanduser()
        if candidate_path.exists() and candidate_path.is_dir():
            logger.info(f"Scanning directory: {candidate_path}")
            valid_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
            for root, _, files in os.walk(candidate_path):
                for f in files:
                    if Path(f).suffix.lower() in valid_exts:
                        targets.append(str(Path(root) / f))
            continue
            
        # Check if playlist
        if _is_youtube_playlist(inp):
            playlist_urls = _extract_playlist_urls(inp)
            targets.extend(playlist_urls)
            continue
            
        # Otherwise treat as URL or single file
        targets.append(inp)
        
    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)
            
    return unique_targets

def process_batch(
    inputs: List[str],
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: str = None,
    mode: str = "api",
    profile: str = "youtube",
    out_dir_base: str = None,
    review_callback=None
) -> Dict[str, Any]:
    """Process a batch of inputs sequentially and aggregate a summary report.
    
    Args:
        inputs: Mixed list of URLs, playlists, or local paths.
        num_clips: Number of shorts to generate per input.
        aspect_ratio: Target aspect ratio (e.g. '9:16').
        download_format: Source resolution (e.g. '720').
        language: Forced language code for transcription (optional).
        mode: Processing mode ('api' or 'local').
        profile: Target export profile (e.g. 'youtube', 'tiktok').
        out_dir_base: Base directory for local outputs.
        review_callback: Optional callback for Human Review Mode.
        
    Returns:
        A dictionary containing processing statistics ('success', 'failed', 'total').
    """
    logger.info(f"Starting batch process for {len(inputs)} input(s)")
    targets = normalize_inputs(inputs)
    logger.info(f"Normalized {len(inputs)} inputs into {len(targets)} unique targets to process.")
    
    report = {
        "total_targets": len(targets),
        "success": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    t0 = time.perf_counter()
    
    for i, target in enumerate(targets, 1):
        logger.info(f"--- Processing target {i}/{len(targets)}: {target} ---")
        try:
            # Generate a subfolder name based on video ID or index
            subfolder = None
            target_out_dir = None
            if out_dir_base:
                vid_id = extract_youtube_video_id(target)
                if vid_id:
                    target_out_dir = os.path.join(out_dir_base, vid_id)
                else:
                    target_out_dir = os.path.join(out_dir_base, f"video_{i:03d}")
            
            result = generate_shorts(
                youtube_url=target,
                num_clips=num_clips,
                aspect_ratio=aspect_ratio,
                download_format=download_format,
                language=language,
                mode=mode,
                profile=profile,
                out_dir_override=target_out_dir,
                review_callback=review_callback
            )
            report["success"] += 1
            report["results"].append({
                "target": target,
                "status": "success",
                "clips_generated": len(result.shorts),
                "source_video": result.source_video_url
            })
        except (ValueError, IOError) as e:
            logger.error(f"IO/Validation Error processing {target}: {e}")
            report["failed"] += 1
            report["errors"].append({"target": target, "error": str(e)})
        except Exception as e:
            logger.error(f"Unexpected error processing {target}: {e}", exc_info=True)
            report["failed"] += 1
            report["errors"].append({"target": target, "error": str(e)})
            
    t1 = time.perf_counter()
    report["time_taken_seconds"] = round(t1 - t0, 2)
    
    if mode == "local" and out_dir_base:
        os.makedirs(out_dir_base, exist_ok=True)
        # Write JSON report
        with open(os.path.join(out_dir_base, "batch_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        # Write MD report
        md_lines = [
            f"# Batch Processing Report",
            f"**Total Targets:** {report['total_targets']}  ",
            f"**Success:** {report['success']}  ",
            f"**Failed:** {report['failed']}  ",
            f"**Time Taken:** {report['time_taken_seconds']}s  ",
            "",
            "## Details"
        ]
        
        for r in report["results"]:
            md_lines.append(f"\n### {r['target']}")
            if r["status"] == "success":
                md_lines.append(f"- **Status:** ✅ Success")
                md_lines.append(f"- **Clips Generated:** {r['clips_generated']}")
                md_lines.append(f"- **Source:** {r['source_video']}")
            else:
                md_lines.append(f"- **Status:** ❌ Failed")
                md_lines.append(f"- **Error:** `{r['error']}`")
                
        with open(os.path.join(out_dir_base, "batch_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
            
    return report
