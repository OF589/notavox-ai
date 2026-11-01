"""Teste la transcription en ligne de commande.

Exemple : python scripts/transcribe.py samples/reunion.mp3 --model small
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.transcription import DEFAULT_MODEL, transcribe  # noqa: E402


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="NotaVox AI — transcription")
    parser.add_argument("audio", help="chemin du fichier audio (mp3, wav, m4a, ogg...)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="tiny, base, small, medium, large-v3 (défaut : %(default)s)")
    parser.add_argument("--language", default=None,
                        help="code de langue (fr, en, ar...) ; vide = détection auto")
    args = parser.parse_args()

    print(f"Transcription de {args.audio} avec le modèle '{args.model}'...")
    result = transcribe(args.audio, model_size=args.model, language=args.language)

    print(f"\nLangue détectée : {result.language} ({result.language_probability:.0%})")
    print(f"Durée audio : {fmt(result.duration)} | traitement : {result.processing_time:.1f} s "
          f"| facteur temps réel : {result.real_time_factor:.2f}\n")
    for s in result.segments:
        print(f"[{fmt(s.start)} → {fmt(s.end)}] {s.text}")

    out = ROOT / "outputs" / f"{Path(args.audio).stem}_transcription.json"
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRésultat enregistré : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
