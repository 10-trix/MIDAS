from flask import Blueprint, request, jsonify

tts_bp = Blueprint("tts", __name__)

@tts_bp.route("/tts", methods=["POST"])
def speak():
    # TODO Week 3: receive text, speak it via pyttsx3
    data = request.get_json()
    text = data.get("text", "")
    return jsonify({"spoken": text})
