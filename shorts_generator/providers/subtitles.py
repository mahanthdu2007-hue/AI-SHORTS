import abc
import os
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict
from ..logger import get_logger
from ..profiles import ExportProfile

logger = get_logger("subtitles")


@dataclass
class SubtitleWord:
    text: str
    start: float
    end: float


@dataclass
class SubtitleEvent:
    words: List[SubtitleWord]
    start: float
    end: float
    margin_v: Optional[int] = None

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


class BaseSubtitleRenderer(abc.ABC):
    """Abstract base class for subtitle rendering."""
    
    @abc.abstractmethod
    def render(self, video_path: str, transcript_data: dict, output_path: str, start_time: float = 0.0, profile: Optional[ExportProfile] = None) -> str:
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
    
    def _build_events(self, transcript_data: dict, start_time: float) -> List[SubtitleEvent]:
        events = []
        segments = transcript_data.get("segments", [])
        
        for seg in segments:
            seg_start = seg.get("start", 0.0) - start_time
            seg_end = seg.get("end", 0.0) - start_time
            
            if seg_end <= 0:
                continue
            seg_start = max(0.0, seg_start)
            
            words = seg.get("words", [])
            if not words:
                continue
                
            chunk = []
            char_count = 0
            line_count = 1
            
            for w_data in words:
                w_str = w_data.get("word", "").strip()
                w_len = len(w_str)
                
                # Check if adding this word exceeds the current line
                if char_count + w_len > 45:
                    if line_count < 2:
                        line_count += 1
                        char_count = w_len + 1
                        chunk.append(SubtitleWord(text=w_str, start=w_data.get("start", 0.0) - start_time, end=w_data.get("end", 0.0) - start_time))
                    else:
                        # Event is full (2 lines). Emit it.
                        if chunk:
                            events.append(SubtitleEvent(
                                words=list(chunk),
                                start=chunk[0].start,
                                end=chunk[-1].end
                            ))
                        chunk = [SubtitleWord(text=w_str, start=w_data.get("start", 0.0) - start_time, end=w_data.get("end", 0.0) - start_time)]
                        char_count = w_len + 1
                        line_count = 1
                else:
                    chunk.append(SubtitleWord(text=w_str, start=w_data.get("start", 0.0) - start_time, end=w_data.get("end", 0.0) - start_time))
                    char_count += w_len + 1
                    
            if chunk:
                events.append(SubtitleEvent(
                    words=list(chunk),
                    start=chunk[0].start,
                    end=chunk[-1].end
                ))
                
        # Fix timing gaps and overlaps
        for i in range(len(events) - 1):
            if events[i+1].start - events[i].end < 0.2:
                events[i].end = events[i+1].start
                
        return events

    def _detect_faces_and_override_margins(self, video_path: str, events: List[SubtitleEvent]) -> None:
        """Sample video and push subtitles up if a face is in the safe zone."""
        try:
            import cv2
        except ImportError:
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Safe zone is bottom 30%
        safe_zone_y = height * 0.70
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # Check 2 frames per second
        frame_step = max(1, int(fps / 2))
        
        face_times = []
        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, w, h) in faces:
                bottom_y = y + h
                if bottom_y > safe_zone_y:
                    time_sec = i / fps
                    face_times.append(time_sec)
                    break
                    
        cap.release()
        
        if not face_times:
            return
            
        # For each event, if its duration overlaps a face time, push it up
        for e in events:
            for ft in face_times:
                if e.start <= ft <= e.end:
                    e.margin_v = int(height * 0.35) # Move up slightly above safe zone
                    break

    def _generate_ass(self, video_path: str, transcript_data: dict, start_time: float, profile: Optional[ExportProfile] = None) -> str:
        events = self._build_events(transcript_data, start_time)
        self._detect_faces_and_override_margins(video_path, events)
        
        if profile is None:
            # Fallback to youtube defaults if no profile provided
            from ..profiles import get_profile
            profile = get_profile("youtube")
            
        # Professional ASS Styling: Rounded Outline (BorderStyle 1, Outline 5, Blur) and Soft Shadow
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{profile.subtitle_font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,{profile.subtitle_margin_l},{profile.subtitle_margin_r},{profile.subtitle_margin_v},1
"""
        
        ass_lines = ["[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"]
        
        for event in events:
            if event.end <= 0:
                continue
            e_start = max(0.0, event.start)
            
            margin_str = str(event.margin_v) if event.margin_v is not None else "0"
            
            for i, w in enumerate(event.words):
                w_start = max(0.0, w.start)
                w_end = w.end
                
                if i < len(event.words) - 1:
                    w_end = event.words[i+1].start
                else:
                    w_end = event.end
                    
                if w_end <= 0:
                    continue
                    
                line_text = ""
                char_count = 0
                for j, wg in enumerate(event.words):
                    word_str = wg.text
                    char_count += len(word_str) + 1
                    
                    if char_count > 45:
                        line_text += "\\N"
                        char_count = len(word_str) + 1
                        
                    if j == i:
                        # Highlight active word in Yellow
                        line_text += f"{{\\c&H00FFFF&}}{word_str}{{\\c&HFFFFFF&}} "
                    else:
                        line_text += f"{word_str} "
                        
                # Strip trailing spaces
                line_text = line_text.strip().replace(" \\N", "\\N")
                
                ass_lines.append(f"Dialogue: 0,{_format_ass_time(w_start)},{_format_ass_time(w_end)},Default,,0,0,{margin_str},,{line_text}")
                
        return header + "\n".join(ass_lines)
        
    def render(self, video_path: str, transcript_data: dict, output_path: str, start_time: float = 0.0, profile: Optional[ExportProfile] = None) -> str:
        logger.info(f"Burning subtitles for {video_path}")
        ass_content = self._generate_ass(video_path, transcript_data, start_time, profile=profile)
        
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
