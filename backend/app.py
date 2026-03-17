from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from routes.sign_to_speech import sign_bp
from routes.audio_to_sign import audio_bp
from routes.tts import tts_bp

app.register_blueprint(sign_bp)
app.register_blueprint(audio_bp)
app.register_blueprint(tts_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
