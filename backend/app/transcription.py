"""Module de transcription multilingue de NotaVox AI (basé sur faster-whisper)."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

DEFAULT_MODEL = os.getenv("NOTAVOX_WHISPER_MODEL", "small")
SAMPLE_RATE = 16_000  # fréquence attendue par Whisper


@dataclass
class Segment:
    start: float  # en secondes
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    duration: float           # durée de l'audio (s)
    processing_time: float    # temps de traitement (s)
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def real_time_factor(self) -> float:
        """< 1 : plus rapide que la durée réelle de l'audio."""
        return self.processing_time / self.duration if self.duration else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["text"] = self.text
        return data


def load_audio(audio_path: Path) -> np.ndarray:
    """Lit un fichier audio (WAV, FLAC, OGG, MP3) en mono 16 kHz."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError(
            f"Format non pris en charge : {audio_path.suffix}. "
            "Utilise un fichier WAV, FLAC, OGG ou MP3."
        ) from exc

    audio = data.mean(axis=1)  # stéréo -> mono
    if sr != SAMPLE_RATE:      # rééchantillonnage simple vers 16 kHz
        n = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio


@lru_cache(maxsize=2)
def load_model(model_size: str = DEFAULT_MODEL) -> WhisperModel:
    """Charge le modèle une seule fois (GPU NVIDIA si disponible, sinon CPU)."""
    return WhisperModel(model_size, device="auto", compute_type="auto")


def transcribe(
    audio_path: str | Path,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcrit un fichier audio.

    language : code ISO ("fr", "en", "ar"...) ou None pour la détection automatique.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {audio_path}")

    model = load_model(model_size)
    start = time.perf_counter()

    audio = load_audio(audio_path)
    segments_iter, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,   # ignore les silences
        beam_size=5,
    )
    # La transcription est paresseuse : elle s'exécute pendant cette boucle.
    segments = [
        Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
        for s in segments_iter
    ]

    return TranscriptionResult(
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=round(info.duration, 2),
        processing_time=round(time.perf_counter() - start, 2),
        segments=segments,
    )
