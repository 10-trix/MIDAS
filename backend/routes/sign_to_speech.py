from flask import Blueprint, request, jsonify

sign_bp = Blueprint("sign_to_speech", __name__)

@sign_bp.route("/sign-to-text", methods=["POST"])
def sign_to_text():
    # TODO Week 3: receive landmark frame, return predicted sign label
    return jsonify({"label": "placeholder", "confidence": 0.0})
