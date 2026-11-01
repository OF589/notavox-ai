# NotaVox AI

Assistant de réunions et de notes vocales multilingue : transcription, résumé, décisions et tâches, dans la langue de votre choix.

> Projet en cours — étape 1 : transcription multilingue.

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation (étape 1)

Placez un fichier audio dans `samples/`, puis :

```bash
python scripts/transcribe.py samples/reunion.mp3
python scripts/transcribe.py samples/reunion.mp3 --model medium --language fr
```

La transcription horodatée s'affiche et est enregistrée en JSON dans `outputs/`.
