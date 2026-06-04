from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging

from app.api.routes.auth import router as auth_router
from app.api.routes.jobs import router as jobs_router
from app.config import settings
from app.services.media.ffmpeg_check import ffmpeg_unavailable_message

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PM Content AI API",
    description="AI content repurposing backend for Pritesh Mody Instagram POC",
    version="1.0.0",
)

origins = settings.cors_origin_list
allow_origin_regex = None

if "*" in origins:
    allow_origin_regex = r"https?://.*"
    origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(jobs_router)


@app.on_event("startup")
async def warn_missing_ffmpeg() -> None:
    msg = ffmpeg_unavailable_message()
    if msg:
        logger.warning("Video processing unavailable: %s", msg)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "PM Content AI API", "docs": "/docs"}
