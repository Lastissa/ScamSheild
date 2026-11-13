"""
main.py - small FastAPI backend for the ScamShield scam-message detector.

Endpoints:
  GET    /health        -> {"status": "ok"}
  POST   /predict       body {"text": "..."}   -> detector JSON
  POST   /predict-batch body {"texts": ["..."]} -> list of detector JSON

CORS is easy to change: edit ALLOWED_ORIGINS below (or set the
CORS_ALLOWED_ORIGINS env var to a comma-separated list).

Run:  uvicorn app.main:app --reload   (from the model/ folder)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import ScamDetector  # noqa: E402

# ---- CORS: change this list (or the env var) to fit your website backend ----
_DEFAULT_ORIGINS = ["*"]   # tighten in production, e.g. ["https://your-site.com"]
_RAW = os.getenv("CORS_ALLOWED_ORIGINS")
ALLOWED_ORIGINS = ([o.strip() for o in _RAW.split(",") if o.strip()]
                   if _RAW else _DEFAULT_ORIGINS)

app = FastAPI(title="ScamShield scam-message detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(TypeError)
def type_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# load the model ONCE at startup
DETECTOR = ScamDetector()


class PredictIn(BaseModel):
    text: str


class PredictBatchIn(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": str(DETECTOR.model_path.name)}


@app.post("/predict")
def predict(inp: PredictIn):
    return DETECTOR.predict(inp.text)


@app.post("/predict-batch")
def predict_batch(inp: PredictBatchIn):
    return DETECTOR.predict_batch(inp.texts)
