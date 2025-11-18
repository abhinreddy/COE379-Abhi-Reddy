import os
import json
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image
import tensorflow as tf

# === Load model and metadata ===
MODEL_PATH = os.path.join("models", "best_model.keras")
METADATA_PATH = os.path.join("models", "metadata.json")

print("[server] Loading model from", MODEL_PATH, flush=True)
MODEL = tf.keras.models.load_model(MODEL_PATH)
print("[server] Model loaded.", flush=True)

metadata = {}
if os.path.exists(METADATA_PATH):
    try:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        print("[server] Loaded metadata:", metadata, flush=True)
    except Exception as exc:
        print("[server] Warning: could not read metadata.json:", exc, flush=True)
else:
    print("[server] metadata.json not found; using defaults.", flush=True)

IMG_HEIGHT = int(metadata.get("img_height", 150))
IMG_WIDTH  = int(metadata.get("img_width", 150))

# === Helpers ===
def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# === HTTP handler ===
class DamageHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/summary":
            info = {
                "model_name": getattr(MODEL, "name", None),
                "input_shape": getattr(MODEL, "input_shape", None),
                "output_shape": getattr(MODEL, "output_shape", None),
                "img_height": IMG_HEIGHT,
                "img_width": IMG_WIDTH,
            }
            if isinstance(metadata, dict):
                for k, v in metadata.items():
                    info[f"meta_{k}"] = v
            self._send_json(info, status=200)
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path == "/inference":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length > 0:
                raw = self.rfile.read(length)
            else:
                raw = self.rfile.read()

            if not raw:
                self._send_json({"error": "Empty request body"}, status=400)
                return

            try:
                arr = preprocess_image(raw)
                preds = MODEL.predict(arr)
                prob_damage = float(preds[0][0])
                label = "damage" if prob_damage >= 0.5 else "no_damage"
                self._send_json({"prediction": label}, status=200)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
        else:
            self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format, *args):
        # Reduce noisy default logging, but keep something
        print("[request]", self.address_string(), "-", format % args, flush=True)

def run():
    port = int(os.environ.get("PORT", "8000"))
    server_address = ("", port)
    httpd = HTTPServer(server_address, DamageHandler)
    print(f"[server] Listening on 0.0.0.0:{port}", flush=True)
    httpd.serve_forever()

if __name__ == "__main__":
    run()
