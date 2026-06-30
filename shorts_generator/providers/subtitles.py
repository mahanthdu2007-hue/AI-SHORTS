import abc
import os
import shutil
import tempfile
import subprocess
from typing import Optional, List, Dict
from ..logger import get_logger

logger = get_logger("subtitles")


class BaseSubtitleRenderer(abc.ABC):
    """Abstract base class for subtitle rendering."""
    
    @abc.abstractmethod
    def render(self, video_path: str, transcript_data: dict, output_path: str, start_time: float = 0.0) -> str:
        """Render subtitles onto the video."""
        pass


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS time format: H:MM:SS.cs"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


class ASSSubtitleRenderer(BaseSubtitleRenderer):
    """Generates professional ASS subtitles and burns them using FFmpeg."""
    
    def _generate_ass(self, transcript_data: dict, start_time: float) -> str:
        # ASS Header with safe margins and styling
        header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,70,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,60,60,350,1
"""
        
        events = ["[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
        segments = transcript_data.get("segments", [])
        
        for seg in segments:
            seg_start = seg.get("start", 0.0) - start_time
            seg_end = seg.get("end", 0.0) - start_time
            
            if seg_end <= 0:
                continue
            seg_start = max(0.0, seg_start)
            
            words = seg.get("words", [])
            
            if not words:
                text = seg.get("text", "").strip()
                events.append(f"Dialogue: 0,{_format_ass_time(seg_start)},{_format_ass_time(seg_end)},Default,,0,0,0,,{text}")
                continue
                
            # Chunk words to ensure max two lines
            word_chunks = []
            chunk = []
            char_count = 0
            for w in words:
                w_str = w.get("word", "").strip()
                if len(chunk) >= 7 or char_count + len(w_str) > 35:
                    word_chunks.append(chunk)
                    chunk = []
                    char_count = 0
                chunk.append(w)
                char_count += len(w_str) + 1
            if chunk:
                word_chunks.append(chunk)

            for chunk_idx, chunk_words in enumerate(word_chunks):
                # find chunk end
                if chunk_idx < len(word_chunks) - 1:
                    chunk_end = word_chunks[chunk_idx+1][0].get("start", 0.0) - start_time
                else:
                    chunk_end = seg_end
                
                for i, w in enumerate(chunk_words):
                    w_start = w.get("start", 0.0) - start_time
                    w_end = w.get("end", 0.0) - start_time
                    
                    # Smooth timing to prevent subtitle flickering
                    if i < len(chunk_words) - 1:
                        next_start = chunk_words[i+1].get("start", 0.0) - start_time
                        if next_start - w_end < 0.2:
                            w_end = next_start
                    else:
                        if chunk_end - w_end < 0.5:
                            w_end = chunk_end
                            
                    w_start = max(0.0, w_start)
                    if w_end <= 0:
                        continue
                        
                    line_text = ""
                    for j, wg in enumerate(chunk_words):
                        word_str = wg.get("word", "").strip()
                        if j == i:
                            # Highlight active word in Yellow
                            line_text += f"{{\\c&H00FFFF&}}{word_str}{{\\c&HFFFFFF&}} "
                        else:
                            line_text += f"{word_str} "
                            
                    line_text = line_text.strip()
                    events.append(f"Dialogue: 0,{_format_ass_time(w_start)},{_format_ass_time(w_end)},Default,,0,0,0,,{line_text}")
                
        return header + "\n".join(events)
        
    def render(self, video_path: str, transcript_data: dict, output_path: str, start_time: float = 0.0) -> str:
        logger.info(f"Burning subtitles for {video_path}")
        ass_content = self._generate_ass(transcript_data, start_time)
        
        fd, ass_path = tempfile.mkstemp(suffix=".ass")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(ass_content)
            
        temp_out = output_path + ".subbed.mp4"
        
        ass_path_esc = os.path.abspath(ass_path).replace('\\', '/')
        ass_path_esc = ass_path_esc.replace(':', r'\:')
        ass_path_esc = ass_path_esc.replace("'", r"\'")
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"ass='{ass_path_esc}'",
            "-c:a", "copy",
            temp_out
        ]
        
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(output_path) and output_path == video_path:
                os.remove(video_path)
            shutil.move(temp_out, output_path)
        except Exception as e:
            logger.error(f"FFmpeg subtitle burn failed: {e}")
            if os.path.exists(temp_out):
                os.remove(temp_out)
            if video_path != output_path:
                shutil.copy2(video_path, output_path)
        finally:
            os.remove(ass_path)
            
        return output_path


def get_subtitle_renderer(mode: str) -> BaseSubtitleRenderer:
    """Factory to get the configured subtitle renderer."""
    return ASSSubtitleRenderer()
