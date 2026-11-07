"""Transcrit une réunion en indiquant qui parle.

Exemple : python scripts/diarize.py samples/reunion_3voix.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.diarization import assign_speakers, diarize  # noqa: E402
from app.transcription import DEFAULT_MODEL, transcribe  # noqa: E402


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NotaVox AI — transcription avec intervenants")
    parser.add_argument("audio", help="fichier audio (wav, mp3, flac, ogg)")
    parser.add_argument("--speakers", type=int, default=None,
                        help="nombre d'intervenants, s'il est connu (sinon détection auto)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle Whisper")
    parser.add_argument("--language", default=None, help="forcer la langue (fr, en, ar, es)")
    args = parser.parse_args()

    print("1/2 Transcription...")
    transcription = transcribe(args.audio, model_size=args.model, language=args.language)

    print("2/2 Reconnaissance des intervenants...")
    start = time.perf_counter()
    turns = diarize(args.audio, num_speakers=args.speakers)
    diarization_time = round(time.perf_counter() - start, 2)

    segments = assign_speakers(transcription.segments, turns)
    speakers = sorted({s["speaker"] for s in segments})

    print(f"\nLangue : {transcription.language} | Intervenants détectés : {len(speakers)}")
    print(f"Temps : transcription {transcription.processing_time:.1f} s, "
          f"intervenants {diarization_time:.1f} s\n")
    for s in segments:
        print(f"[{fmt(s['start'])} → {fmt(s['end'])}] {s['speaker']} : {s['text']}")

    # Le texte avec les noms des intervenants sert ensuite à analyze.py
    text = "\n".join(f"{s['speaker']} : {s['text']}" for s in segments)
    data = {
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "duration": transcription.duration,
        "processing_time": transcription.processing_time,
        "diarization_time": diarization_time,
        "speakers": speakers,
        "segments": segments,
        "text": text,
    }
    out = ROOT / "outputs" / f"{Path(args.audio).stem}_transcription.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultat enregistré : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
