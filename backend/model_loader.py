"""Loads the pretrained model ONCE and runs inference. No training happens here."""
import logging

import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from . import config

log = logging.getLogger("esc")


class SoundClassifier:
    def __init__(self, model_id: str = config.MODEL_ID):
        self.model_id = model_id
        log.info("Loading pretrained model '%s' (first run downloads ~350 MB)...", model_id)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_id)
        self.model = ASTForAudioClassification.from_pretrained(model_id)
        self.model.eval()  # inference mode: weights are never updated
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}
        log.info("Model ready with %d classes.", len(self.labels))

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def predict(self, features, top_k: int = config.TOP_K):
        with torch.inference_mode():
            logits = self.model(**features).logits[0]
        # AudioSet is multi-label: each class gets an independent 0-1 score (sigmoid)
        scores = torch.sigmoid(logits)
        values, indices = torch.topk(scores, k=top_k)
        return [
            {"label": self.labels[int(i)], "score": round(float(v), 4)}
            for v, i in zip(values, indices)
        ]


_classifier = None


def load_classifier() -> SoundClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SoundClassifier()
    return _classifier


def get_classifier() -> SoundClassifier:
    if _classifier is None:
        raise RuntimeError("Model not loaded yet.")
    return _classifier
