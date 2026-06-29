import os
import subprocess
from typing import List
from pathlib import Path

from .models import Highlight
from .logger import get_logger

logger = get_logger("review")

def _extract_thumbnail(source_video_url: str, start_time: float, output_path: str):
    """Extract a single frame thumbnail using ffmpeg."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time),
        "-i", source_video_url,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to extract thumbnail for {output_path}: {e}")

def cli_human_review_callback(clips: List[Highlight], source_video_url: str) -> List[Highlight]:
    """CLI interactive human review loop."""
    print("\n" + "="*50)
    print(" HUMAN REVIEW MODE ")
    print("="*50)
    
    cache_dir = os.path.join(os.getcwd(), ".cache", "thumbnails")
    
    # Pre-extract thumbnails
    for i, c in enumerate(clips, 1):
        thumb_path = os.path.join(cache_dir, f"thumb_{i}.jpg")
        _extract_thumbnail(source_video_url, c.start_time, thumb_path)
        c._thumb_path = thumb_path  # temporary attribute
        
    working_clips = list(clips)
    
    while True:
        print("\n--- CURRENT CLIPS ---")
        for i, c in enumerate(working_clips, 1):
            print(f"[{i}] {c.title}")
            print(f"    Start: {c.start_time:.1f}s | End: {c.end_time:.1f}s")
            thumb = getattr(c, '_thumb_path', 'None')
            print(f"    Thumbnail: {thumb}")
            if c.criteria:
                print(f"    Viral Score: {c.criteria.total} | Hook: {c.criteria.hook_strength} | Story: {c.criteria.storytelling} | Retention: {c.criteria.viewer_retention_prediction}")
            else:
                print(f"    Score: {c.score}")
            print(f"    Hook: {c.hook_sentence}")
            print("-" * 30)
            
        print("\nCommands:")
        print("  accept <id1,id2,...> (or 'accept all')")
        print("  reject <id1,id2,...>")
        print("  trim <id> <new_start> <new_end>")
        print("  move <id> <new_index> (1-based index)")
        print("  done (finish review and proceed to rendering)")
        
        cmd = input("Review action> ").strip().lower()
        if not cmd:
            continue
            
        parts = cmd.split()
        action = parts[0]
        
        if action == "done":
            break
            
        elif action == "accept":
            if len(parts) > 1:
                if parts[1] == "all":
                    break
                else:
                    try:
                        ids = [int(x.strip()) for x in parts[1].split(',')]
                        working_clips = [c for i, c in enumerate(working_clips, 1) if i in ids]
                    except ValueError:
                        print("Invalid IDs.")
                        
        elif action == "reject":
            if len(parts) > 1:
                try:
                    ids = [int(x.strip()) for x in parts[1].split(',')]
                    working_clips = [c for i, c in enumerate(working_clips, 1) if i not in ids]
                except ValueError:
                    print("Invalid IDs.")
                    
        elif action == "trim":
            if len(parts) >= 4:
                try:
                    cid = int(parts[1])
                    new_start = float(parts[2])
                    new_end = float(parts[3])
                    if 1 <= cid <= len(working_clips):
                        working_clips[cid-1].start_time = new_start
                        working_clips[cid-1].end_time = new_end
                        print(f"Trimmed clip {cid} to {new_start}-{new_end}.")
                except ValueError:
                    print("Invalid format. Use: trim <id> <start> <end>")
                    
        elif action == "move":
            if len(parts) >= 3:
                try:
                    cid = int(parts[1])
                    new_idx = int(parts[2])
                    if 1 <= cid <= len(working_clips) and 1 <= new_idx <= len(working_clips):
                        c = working_clips.pop(cid - 1)
                        working_clips.insert(new_idx - 1, c)
                        print(f"Moved clip {cid} to position {new_idx}.")
                except ValueError:
                    print("Invalid format. Use: move <id> <new_index>")
        else:
            print("Unknown command.")
            
    print("Human Review Complete. Proceeding to rendering...")
    return working_clips
