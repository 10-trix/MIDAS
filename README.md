# Sign Language Translator

A two-way sign language translation system:
- **Sign → Speech**: webcam detects hand signs, speaks them aloud
- **Audio → Sign**: audio/video input transcribed and shown as MediaPipe avatar

## Quick Start
```bash
chmod +x setup.sh && ./setup.sh
source venv/bin/activate
python scripts/test_mediapipe.py
```

## Structure
| Folder | Purpose |
|---|---|
| `backend/` | Flask API — shared by desktop app and extension |
| `data/` | Datasets, pose DB, gloss maps |
| `models/` | Trained classifier `.pkl` files |
| `desktop/` | PyQt5 standalone app |
| `extension/` | Chrome Manifest V3 extension |
| `notebooks/` | Jupyter — training and evaluation |
| `scripts/` | Utility scripts — data collection, testing |

## Team
| Member | Role |
|---|---|
| M1 | Sign→Speech, classifier, desktop UI |
| M2 | Whisper STT, NLP gloss mapper |
| M3 | Flask API, pose renderer, interpolator |
| M4 | Chrome extension, UI polish, report |
