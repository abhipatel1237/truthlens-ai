from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os
import tempfile

load_dotenv()

app = Flask(__name__)
@app.route("/")
def home():
    return send_file("index.html")
    @app.route("/script.js")
def script():
    return send_file("script.js", mimetype="application/javascript")

@app.route("/style.css")
def style():
    return send_file("style.css", mimetype="text/css")
CORS(app)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN .env file me nahi mila")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

MODEL_ID = "haywoodsloan/ai-image-detector-deploy"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "TruthLens backend is running"
    })


@app.route("/analyze", methods=["POST"])
def analyze():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not file.content_type or not file.content_type.startswith("image/"):
        return jsonify({
            "error": "Abhi sirf images supported hain"
        }), 400

    temp_path = None

    try:
        extension = os.path.splitext(file.filename)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            file.save(temp_file)
            temp_path = temp_file.name

        result = client.image_classification(
            temp_path,
            model=MODEL_ID
        )

        predictions = [
            {
                "label": item.label,
                "score": round(float(item.score), 4)
            }
            for item in result
        ]

        return jsonify({
            "success": True,
            "predictions": predictions
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )