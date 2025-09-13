import os

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from common.log import logger

app = FastAPI(
    title="Feifei Deep Research API",
    description="API for Feifei",
    version="0.1.0",
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Use the configured list of methods
    allow_headers=["*"],  # Now allow all headers, but can be restricted further
)
