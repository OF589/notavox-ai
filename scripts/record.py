"""Enregistre une note vocale depuis le micro.
Exemple : python scripts/record.py samples/note_fr.wav
"""
import argparse
import queue
from pathlib import Path

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000


def main() -> None:
    parser = argparse.ArgumentParser(description="NotaVox AI — enregistrement audio")
    parser.add_argument("output", help="ex : samples/note_fr.wav")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(status)
        frames.put(indata.copy())

    with sf.SoundFile(out, mode="w", samplerate=SAMPLE_RATE, channels=1) as file:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
            input("🎙️  Enregistrement en cours... appuie sur Entrée pour arrêter.\n")
        while not frames.empty():
            file.write(frames.get())

    print(f"✅ Enregistré : {out} ({sf.info(out).duration:.0f} secondes)")


if __name__ == "__main__":
    main()
