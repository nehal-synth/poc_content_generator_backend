from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    UPLOADED = "uploaded"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    CLASSIFYING = "classifying"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentIntent(BaseModel):
    recipe: bool = False
    storytelling: bool = False
    education: bool = False
    case_study: bool = False
    opinion: bool = False


class Entities(BaseModel):
    people: list[str] = Field(default_factory=list)
    brands: list[str] = Field(default_factory=list)
    restaurants: list[str] = Field(default_factory=list)
    hotels: list[str] = Field(default_factory=list)
    cocktails: list[str] = Field(default_factory=list)
    ingredients: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)


class ContentAnalysis(BaseModel):
    transcript: str = ""
    visual_description: str = ""
    summary: str = ""
    entities: Entities = Field(default_factory=Entities)
    topics: list[str] = Field(default_factory=list)
    intent: ContentIntent = Field(default_factory=ContentIntent)
    audio_visual_divergence: bool = False
    audio_focus_summary: str = ""
    visual_focus_summary: str = ""
    repurposing_guidance: str = ""


class FormatScore(BaseModel):
    format: str
    confidence: int
    description: str = ""


class Classification(BaseModel):
    pillar: str
    pillar_confidence: int
    pillar_scores: dict[str, int] = Field(default_factory=dict)
    format: str
    format_confidence: int
    format_scores: list[FormatScore] = Field(default_factory=list)


class ReelScript(BaseModel):
    hook: str = ""
    body: str = ""
    cta: str = ""


class Caption(BaseModel):
    hook: str = ""
    story: str = ""
    recipe: str = ""
    cta: str = ""
    hashtags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CarouselSlides(BaseModel):
    slide_1: str = ""
    slide_2: str = ""
    slide_3: str = ""
    slide_4: str = ""
    slide_5: str = ""


class StoryIdea(BaseModel):
    type: str
    content: str


class GeneratedAssets(BaseModel):
    reel: ReelScript = Field(default_factory=ReelScript)
    caption: Caption = Field(default_factory=Caption)
    carousel: CarouselSlides = Field(default_factory=CarouselSlides)
    stories: list[StoryIdea] = Field(default_factory=list)


class PublishingRecommendation(BaseModel):
    platform: str = "Instagram"
    content_type: str = "Reel"
    recommended_day: str = ""
    recommended_time: str = ""
    series: str = ""
    reason: str = ""
    confidence: int = 0
    clearance_flag: str = ""
    follow_up_slots: list[dict[str, str]] = Field(default_factory=list)


class JobFile(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    is_video: bool


# ---------------------------------------------------------------------------
# Cost & usage metrics — backend-only, never sent to the frontend UI.
# ---------------------------------------------------------------------------

class WhisperUsage(BaseModel):
    duration_seconds: float = 0.0
    cost: float = 0.0


class LLMStepUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0


class CostAndUsageMetrics(BaseModel):
    whisper: WhisperUsage = Field(default_factory=WhisperUsage)
    vision_analysis: LLMStepUsage = Field(default_factory=LLMStepUsage)
    transcript_correction: LLMStepUsage = Field(default_factory=LLMStepUsage)
    classification: LLMStepUsage = Field(default_factory=LLMStepUsage)
    generation: LLMStepUsage = Field(default_factory=LLMStepUsage)
    total_cumulative_cost: float = 0.0


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    status_message: str = ""
    error: str | None = None
    file: JobFile
    companion_text: str = ""
    analysis: ContentAnalysis | None = None
    classification: Classification | None = None
    assets: GeneratedAssets | None = None
    recommendation: PublishingRecommendation | None = None
    created_at: str
    updated_at: str
    processing_seconds: float | None = None
    profiling_metrics: Dict[str, float] = Field(default_factory=dict)
    # Backend-only cost/token tracking — excluded from public API responses
    cost_and_usage_metrics: CostAndUsageMetrics = Field(default_factory=CostAndUsageMetrics)


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    filename: str
    pillar: str | None = None
    format: str | None = None
    created_at: str


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int


class HealthResponse(BaseModel):
    status: str
    openai_configured: bool
