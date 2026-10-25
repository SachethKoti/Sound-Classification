"""Central settings. Change values here (or with environment variables)."""
import os

# Pretrained model from the Hugging Face Hub (downloaded automatically the first time).
# Audio Spectrogram Transformer fine-tuned on AudioSet (527 sound classes,
# includes dog, rain, siren, engine, thunderstorm, fireworks, ...).
MODEL_ID = os.getenv("MODEL_ID", "MIT/ast-finetuned-audioset-10-10-0.4593")

TARGET_SAMPLE_RATE = 16000        # the model expects 16 kHz mono audio
WINDOW_SECONDS = 10.0             # the model looks at ~10 seconds at a time
MIN_SECONDS = 0.5                 # shorter clips are rejected
MAX_UPLOAD_MB = 25
MAX_DECODE_SECONDS = 600          # refuse files longer than 10 minutes
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
SILENCE_RMS_THRESHOLD = 1e-4
LOW_CONFIDENCE_THRESHOLD = 0.25   # below this we warn "may be outside supported classes"
TOP_K = 3
