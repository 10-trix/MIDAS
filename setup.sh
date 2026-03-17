#!/bin/bash
echo "Setting up Sign Language Translator..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python3 -c "import whisper; whisper.load_model('tiny')"
echo ""
echo "Done! Run: source venv/bin/activate"
echo "Then:      python scripts/test_mediapipe.py"
