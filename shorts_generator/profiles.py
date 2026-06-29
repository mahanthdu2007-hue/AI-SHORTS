from pydantic import BaseModel, Field

class ExportProfile(BaseModel):
    name: str = Field(..., description="Name of the platform profile")
    resolution: str = Field("1080x1920", description="Target output resolution (e.g., 1080x1920)")
    bitrate: str = Field("10M", description="Video bitrate (e.g., 10M, 8M, 15M)")
    fps: int = Field(30, description="Target framerate (30 or 60)")
    subtitle_font_size: int = Field(75, description="Font size for subtitles")
    subtitle_margin_v: int = Field(250, description="Vertical margin from bottom for subtitles")
    subtitle_margin_l: int = Field(15, description="Left margin for subtitles")
    subtitle_margin_r: int = Field(15, description="Right margin for subtitles")
    output_suffix: str = Field("", description="Suffix to append to final file (e.g., _tiktok)")
    codec: str = Field("libx264", description="Video codec to use")


PROFILES = {
    "youtube": ExportProfile(
        name="youtube",
        resolution="1080x1920",
        bitrate="10M",
        fps=30,
        subtitle_font_size=75,
        subtitle_margin_v=250,
        subtitle_margin_l=15,
        subtitle_margin_r=15,
        output_suffix="_youtube",
        codec="libx264"
    ),
    "tiktok": ExportProfile(
        name="tiktok",
        resolution="1080x1920",
        bitrate="12M",
        fps=60,
        subtitle_font_size=70,
        subtitle_margin_v=400,  # TikTok has a high description box
        subtitle_margin_l=25,
        subtitle_margin_r=90,  # Right side icons
        output_suffix="_tiktok",
        codec="libx264"
    ),
    "instagram": ExportProfile(
        name="instagram",
        resolution="1080x1920",
        bitrate="8M", # Instagram compresses heavily
        fps=30,
        subtitle_font_size=75,
        subtitle_margin_v=300,
        subtitle_margin_l=15,
        subtitle_margin_r=40,
        output_suffix="_instagram",
        codec="libx264"
    ),
}

def get_profile(name: str) -> ExportProfile:
    """Retrieve an export profile by name. Defaults to 'youtube' if not found."""
    profile_name = name.lower()
    return PROFILES.get(profile_name, PROFILES["youtube"])
