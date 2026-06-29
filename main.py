"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
        
    python main.py "https://youtube.com/playlist?list=..." "/path/to/local/folder"
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts
from shorts_generator.batch import process_batch
from shorts_generator.review import cli_human_review_callback


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("urls", nargs="+", help="YouTube URLs, file:// URLs, local file paths, playlists, or folders")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--profile", default="youtube", choices=["youtube", "tiktok", "instagram"], help="Export profile to format for a specific platform")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument("--human-review", action="store_true", help="Pause pipeline to allow human review of clips before rendering")
    args = parser.parse_args()

    try:
        from shorts_generator.config import LOCAL_OUTPUT_DIR
        out_dir_base = LOCAL_OUTPUT_DIR if args.mode == "local" else None
        
        callback = cli_human_review_callback if args.human_review else None
        
        report = process_batch(
            inputs=args.urls,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
            profile=args.profile,
            out_dir_base=out_dir_base,
            review_callback=callback
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Batch Processing Complete")
    print(f"Total Targets: {report['total_targets']}")
    print(f"Success:       {report['success']}")
    print(f"Failed:        {report['failed']}")
    print(f"Time Taken:    {report['time_taken_seconds']}s")
    print("=" * 72)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull batch JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
