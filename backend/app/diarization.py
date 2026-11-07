"""Reconnaissance des intervenants (qui parle, et quand) avec pyannote.audio."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

from app.transcription import SAMPLE_RATE, Segment, load_audio

load_dotenv()  # lit HF_TOKEN dans le fichier .env
DIARIZATION_MODEL = os.getenv(
    "NOTAVOX_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Charge le modèle pyannote une seule fois."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN est introuvable : vérifie le fichier .env.")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:  # anciennes versions de pyannote.audio
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            "Accès au modèle refusé : accepte ses conditions sur Hugging Face."
        )
    return pipeline


def diarize(audio_path: str | Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
    """Découpe l'audio en tours de parole : Intervenant 1, Intervenant 2..."""
    audio = load_audio(Path(audio_path))
    # Audio passé en mémoire : pas besoin de FFmpeg pour le décoder
    waveform = torch.from_numpy(audio).unsqueeze(0)
    options = {"num_speakers": num_speakers} if num_speakers else {}
    output = load_pipeline()({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **options)

    # pyannote 4 : version "exclusive" = une seule personne à la fois, idéale pour la transcription
    annotation = output
    for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
        if hasattr(output, attribute):
            annotation = getattr(output, attribute)
            break

    names: dict[str, str] = {}
    turns = []
    for turn, _, label in annotation.itertracks(yield_label=True):
        name = names.setdefault(label, f"Intervenant {len(names) + 1}")
        turns.append(SpeakerTurn(round(turn.start, 2), round(turn.end, 2), name))
    return turns


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[dict]:
    """Associe chaque phrase transcrite à l'intervenant qui parle le plus pendant celle-ci."""
    result = []
    for seg in segments:
        overlaps: dict[str, float] = {}
        for t in turns:
            overlap = min(seg.end, t.end) - max(seg.start, t.start)
            if overlap > 0:
                overlaps[t.speaker] = overlaps.get(t.speaker, 0.0) + overlap
        speaker = max(overlaps, key=overlaps.get) if overlaps else "Inconnu"
        result.append({"start": seg.start, "end": seg.end, "speaker": speaker, "text": seg.text})

    # Renumérote les intervenants dans l'ordre où ils prennent la parole
    order: dict[str, str] = {}
    for item in result:
        if item["speaker"] != "Inconnu":
            order.setdefault(item["speaker"], f"Intervenant {len(order) + 1}")
            item["speaker"] = order[item["speaker"]]
    return result
