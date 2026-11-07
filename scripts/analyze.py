"""Analyse une transcription : résumé, décisions et tâches.

Exemple : python scripts/analyze.py outputs/tts_fr_transcription.json --lang fr
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.analysis import DEFAULT_MODEL, analyze  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="NotaVox AI — analyse de réunion")
    parser.add_argument("transcription", help="fichier JSON produit par transcribe.py")
    parser.add_argument("--lang", default="fr",
                        help="langue du compte rendu : fr, en, ar, es (défaut : fr)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="modèle Ollama (défaut : %(default)s)")
    args = parser.parse_args()

    source = Path(args.transcription)
    transcript = json.loads(source.read_text(encoding="utf-8"))["text"]

    print(f"Analyse avec '{args.model}', compte rendu en '{args.lang}'...\n")
    result = analyze(transcript, output_language=args.lang, model=args.model)

    print("📝 RÉSUMÉ")
    print(result.summary, "\n")
    print("📌 POINTS CLÉS")
    for p in result.key_points:
        print(f"  • {p}")
    print("\n✅ DÉCISIONS")
    for d in result.decisions or ["(aucune)"]:
        print(f"  • {d}")
    print("\n📋 TÂCHES")
    for t in result.tasks or []:
        print(f"  • {t.task} — {t.owner} — {t.deadline}")
    if not result.tasks:
        print("  • (aucune)")
    print(f"\n⏱️  Temps d'analyse : {result.processing_time:.1f} s")

    stem = source.stem.replace("_transcription", "")
    out = ROOT / "outputs" / f"{stem}_analyse_{args.lang}.json"
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Résultat enregistré : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
