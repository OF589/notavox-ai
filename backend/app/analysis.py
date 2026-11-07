"""Analyse d'une transcription par un LLM : résumé, points clés, décisions et tâches."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import requests

OLLAMA_URL = os.getenv("NOTAVOX_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("NOTAVOX_LLM_MODEL", "qwen2.5:3b")

LANGUAGES = {
    "fr": "French",
    "en": "English",
    "ar": "Modern Standard Arabic",
    "es": "Spanish",
}

# Structure imposée au LLM (Ollama garantit un JSON conforme à ce schéma)
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "deadline": {"type": "string"},
                },
                "required": ["task", "owner", "deadline"],
            },
        },
    },
    "required": ["summary", "key_points", "decisions", "tasks"],
}

SYSTEM_PROMPT = """You are NotaVox AI, an assistant that writes meeting minutes.
You receive the transcript of a meeting or a voice note. It may contain
transcription errors, hesitations and repetitions: interpret them sensibly.

Rules:
- Use ONLY information present in the transcript. Never invent people, dates or tasks.
- summary: 2 to 4 sentences describing the meeting.
- key_points: the main topics discussed.
- decisions: what was decided. Empty list if nothing was decided.
- tasks: concrete actions someone must do. "owner" is the person responsible and
  "deadline" the due date as said in the meeting; if not mentioned, write the word
  "unspecified" translated into the output language.
- Write ALL text in {language}, whatever the language of the transcript.
- Answer with JSON only."""


@dataclass
class Task:
    task: str
    owner: str
    deadline: str


@dataclass
class MeetingAnalysis:
    language: str
    model: str
    processing_time: float
    summary: str
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(
    transcript: str,
    output_language: str = "fr",
    model: str = DEFAULT_MODEL,
) -> MeetingAnalysis:
    """Envoie la transcription au LLM et renvoie une analyse structurée."""
    if not transcript.strip():
        raise ValueError("La transcription est vide.")
    language = LANGUAGES.get(output_language, output_language)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(language=language)},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0.2},  # réponses stables et factuelles
    }

    start = time.perf_counter()
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Impossible de joindre Ollama. Vérifie que l'application Ollama est lancée."
        ) from exc

    data = json.loads(response.json()["message"]["content"])
    return MeetingAnalysis(
        language=output_language,
        model=model,
        processing_time=round(time.perf_counter() - start, 2),
        summary=data["summary"],
        key_points=data["key_points"],
        decisions=data["decisions"],
        tasks=[Task(**t) for t in data["tasks"]],
    )
