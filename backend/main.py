"""FastAPI app: serves the frontend and the /api/predict endpoint."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from . import config
from .audio_preprocessing import (
    AudioError, check_audio_quality, extract_features,
    load_audio, select_window, validate_upload,
)
from .model_loader import get_classifier, load_classifier

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("esc")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_classifier()  # load the model once at startup
    yield


app = FastAPI(title="Environmental Sound Classifier", lifespan=lifespan)


@app.get("/api/health")
def health():
    clf = get_classifier()
    return {"status": "ok", "model": clf.model_id, "classes": clf.num_classes}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    try:
        data = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        ext = validate_upload(file.filename, data)           # 1. validate
        waveform = load_audio(data, ext)                      # 2. decode, mono, 16 kHz
        duration = check_audio_quality(waveform)              # 3. empty/silent/short checks
        segment, start, length = select_window(waveform)      # 4. choose 10 s window
        clf = get_classifier()
        features = extract_features(segment, clf.feature_extractor)  # 5. features
        top = clf.predict(features)                           # 6. pretrained inference
    except AudioError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Something went wrong while processing the audio.")

    best = top[0]
    warning = None
    if best["score"] < config.LOW_CONFIDENCE_THRESHOLD:
        warning = ("Low confidence: this sound may be outside the classes the model "
                   "knows, or the recording is unclear / mixed. Treat the result with caution.")
    return {
        "predicted_label": best["label"],
        "confidence": best["score"],
        "top_predictions": top,
        "duration_seconds": round(duration, 2),
        "analyzed_window": {"start": round(start, 2), "length": round(length, 2)},
        "warning": warning,
        "model": clf.model_id,
        "num_classes": clf.num_classes,
    }


# Must be last: serves index.html, style.css, app.js
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
