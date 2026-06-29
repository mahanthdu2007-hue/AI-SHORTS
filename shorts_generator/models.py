from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from .config import VIRAL_SCORE_WEIGHTS


class TranscriptWord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    start: float
    end: float
    text: str
    words: Optional[List[TranscriptWord]] = None


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    duration: float
    segments: List[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(f"[{s.start:.1f}s] {s.text.strip()}" for s in self.segments)


class ScoreCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hook_strength: int = Field(..., description="0-10 score for hook strength")
    curiosity_gap: int = Field(..., description="0-10 score for curiosity gap")
    emotional_impact: int = Field(..., description="0-10 score for emotional impact")
    storytelling: int = Field(..., description="0-10 score for storytelling")
    educational_value: int = Field(..., description="0-10 score for educational value")
    entertainment_value: int = Field(..., description="0-10 score for entertainment value")
    conflict: int = Field(..., description="0-10 score for conflict")
    surprise: int = Field(..., description="0-10 score for surprise")
    authority_credibility: int = Field(..., description="0-10 score for authority/credibility")
    viewer_retention_prediction: int = Field(..., description="0-10 score for viewer retention prediction")
    shareability: int = Field(..., description="0-10 score for shareability")
    comment_potential: int = Field(..., description="0-10 score for comment potential")
    story_completion_score: int = Field(0, description="0-10 score for complete story arc (Setup to Resolution)")
    emotion_score: int = Field(0, description="0-10 score for emotional energy and audio cues (excitement, laughter, surprise) vs monotone")

    @property
    def total(self) -> int:
        score = 0.0
        score += self.hook_strength * VIRAL_SCORE_WEIGHTS.get("hook_strength", 1.0)
        score += self.curiosity_gap * VIRAL_SCORE_WEIGHTS.get("curiosity_gap", 1.0)
        score += self.emotional_impact * VIRAL_SCORE_WEIGHTS.get("emotional_impact", 1.0)
        score += self.storytelling * VIRAL_SCORE_WEIGHTS.get("storytelling", 1.0)
        score += self.educational_value * VIRAL_SCORE_WEIGHTS.get("educational_value", 1.0)
        score += self.entertainment_value * VIRAL_SCORE_WEIGHTS.get("entertainment_value", 1.0)
        score += self.conflict * VIRAL_SCORE_WEIGHTS.get("conflict", 1.0)
        score += self.surprise * VIRAL_SCORE_WEIGHTS.get("surprise", 1.0)
        score += self.authority_credibility * VIRAL_SCORE_WEIGHTS.get("authority_credibility", 1.0)
        score += self.viewer_retention_prediction * VIRAL_SCORE_WEIGHTS.get("viewer_retention_prediction", 1.0)
        score += self.shareability * VIRAL_SCORE_WEIGHTS.get("shareability", 1.0)
        score += self.comment_potential * VIRAL_SCORE_WEIGHTS.get("comment_potential", 1.0)
        score += self.story_completion_score * VIRAL_SCORE_WEIGHTS.get("story_completion_score", 1.0)
        score += self.emotion_score * VIRAL_SCORE_WEIGHTS.get("emotion_score", 1.0)
        return int(round(score))


class ShortMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(..., description="Optimized YouTube Shorts title")
    thumbnail_text: str = Field(..., description="Catchy text for custom thumbnail")
    description: str = Field(..., description="YouTube Shorts description with dynamic CTA")
    hashtags: List[str] = Field(..., description="List of hashtags")
    keywords: List[str] = Field(..., description="List of keywords for backend tags")
    hook_sentence: str = Field(..., description="The hook sentence")
    virality_explanation: str = Field(..., description="Explanation of why it will go viral")


class Highlight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = Field(..., description="Catchy title for the highlight")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    score: int = Field(0, description="Total viral potential score (max 100)")
    hook_sentence: str = Field(..., description="The opening line that grabs attention")
    virality_reason: str = Field(..., description="Why this clip is viral")
    story_arc_stages: Optional[List[str]] = Field(default=None, description="Stages of the story arc detected (e.g. Setup, Conflict, Rising Action, Peak, Resolution)")
    criteria: Optional[ScoreCriteria] = None
    rejection_reason: Optional[str] = Field(None, description="Reason if the clip is rejected")
    clip_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[ShortMetadata] = Field(default=None, description="Generated metadata for the Short")
    quality_report: Optional[dict] = Field(default=None, description="Final quality diagnostics report")


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    source_video_url: str
    transcript: Transcript
    highlights: List[Highlight]
    shorts: List[Highlight]
