
# Environmental Sound Classifier (pretrained model, no training)

Upload an audio file -> validate -> preprocess -> pretrained model -> predicted sound,
confidence and top-3 predictions.

## The model
**Audio Spectrogram Transformer (AST)**, `MIT/ast-finetuned-audioset-10-10-0.4593`, from Hugging Face.
Pretrained and fine-tuned by its authors on **AudioSet** (527 sound classes: dog, bark, rain,
thunderstorm, siren, engine, fireworks, speech, music...). We only run inference; nothing is trained.

* Input: 16 kHz mono audio, about 10 seconds (the feature extractor pads/crops to 1024 frames
  of 128-bin log-mel filterbank features).
* Output: one score per class (0-1). AudioSet is multi-label, so scores are independent and
  do not sum to 100%.
* Limits: it only knows its 527 classes. A low top score (< 25%) triggers an
  "may be outside the supported classes" warning in the app.
* It is a broader dataset than ESC-50 (50 classes); ESC-50 sounds like dog bark, rain, siren,
  engine, thunderstorm, fireworks are all covered by AudioSet labels.

## Project structure
```
esc_app/
  requirements.txt
  backend/
    config.py               settings (model id, sample rate, limits)
    audio_preprocessing.py  validation, decoding, mono, 16 kHz resample, window, features
    model_loader.py         loads the pretrained model once, runs inference
    main.py                 FastAPI app + /api/predict, serves the frontend
  frontend/
    index.html  style.css  app.js
```

## Setup (step by step)
1. Install **Python 3.9 - 3.12** from python.org (tick "Add Python to PATH" on Windows).
2. (Recommended) Install **FFmpeg** so MP3/M4A/other formats always decode:
   * Windows: `winget install ffmpeg`   * macOS: `brew install ffmpeg`   * Ubuntu: `sudo apt install ffmpeg`
3. Open a terminal inside the `esc_app` folder and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (macOS / Linux)
   ```
4. Install dependencies (PyTorch is large, this takes a few minutes):
   ```
   pip install -r requirements.txt
   ```
5. Start the server:
   ```
   uvicorn backend.main:app --reload
   ```
   The **first start downloads the model (~350 MB, needs internet)**, then it is cached.
   Wait for the log line `Model ready with 527 classes.`
6. Open **http://127.0.0.1:8000** in your browser, upload a sound, press **Predict Sound**.

API docs are at http://127.0.0.1:8000/docs . Health check: `/api/health`.

## Troubleshooting
* "Could not decode this audio file": install FFmpeg (step 2) and restart.
* Port in use: `uvicorn backend.main:app --port 8001`.
* Use a different pretrained model: set the environment variable `MODEL_ID` to another
  AST audio-classification checkpoint from the Hugging Face Hub (e.g. one fine-tuned on ESC-50).
  The code reads labels from the model, so no other changes are needed.

# Sound-Classification
predicts the sound based on the input sound file
 eae49893f5cb182c11733e82a8358b64c9eb71ae
