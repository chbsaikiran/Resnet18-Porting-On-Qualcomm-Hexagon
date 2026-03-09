import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get(
    "ONNX_MODEL", os.path.join("..", "Training", "resnet18_cifar10.onnx")
)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
IMG_SIZE = 32

session = ort.InferenceSession(MODEL_PATH)
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess(file_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD

    img = np.transpose(img, (2, 0, 1))
    return img[np.newaxis, ...]


def predict(file_bytes: bytes):
    tensor = preprocess(file_bytes)
    logits = session.run([output_name], {input_name: tensor})[0][0]

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    top5_idx = probs.argsort()[::-1][:5]
    results = [
        {"label": CLASS_NAMES[i], "confidence": f"{probs[i] * 100:.1f}%"}
        for i in top5_idx
    ]
    return results


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify(error="No image uploaded"), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="Empty filename"), 400

    file_bytes = file.read()
    results = predict(file_bytes)
    return jsonify(predictions=results)


if __name__ == "__main__":
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)
