from flask import Blueprint, request, jsonify

audio_bp = Blueprint("audio_to_sign", __name__)

@audio_bp.route("/audio-to-signs", methods=["POST"])
def audio_to_signs():
    # TODO Week 3: receive audio blob, return list of sign pose frames
    return jsonify({"signs": [], "transcript": ""})
