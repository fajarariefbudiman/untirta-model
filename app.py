from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io
import time
from PIL import Image

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD YOLO MODELS
# ============================================================
def load_yolo_model(path: str, label: str):
    try:
        print(f"[INFO] Loading YOLO {label} model...")
        model = YOLO(path)
        print(f"[SUCCESS] YOLO {label} Model loaded: {path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed loading YOLO {label} Model: {e}")
        return None


letter_model = load_yolo_model("best.pt", "LETTER")
sentence_model = load_yolo_model("best_sentence.pt", "SENTENCE")
object_model = load_yolo_model("best_object.pt", "OBJECT")


# ============================================================
# REQUEST BODY MODEL
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    print("[DEBUG] Root endpoint accessed")
    return {"message": "Server SIBI Aktif (Huruf + Kata + Object)"}


# ============================================================
# HELPER: Base64 to OpenCV Image
# ============================================================
def decode_base64_image(b64_string):
    try:
        print("[DEBUG] Starting Base64 decode...")

        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        image_bytes = base64.b64decode(b64_string)
        print(f"[DEBUG] Base64 decoded size: {len(image_bytes)} bytes")

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        print(f"[DEBUG] Image converted to OpenCV format: shape={img.shape}")
        return img

    except Exception as e:
        print(f"[ERROR] Failed decoding Base64: {e}")
        raise


# ============================================================
# YOLO INFERENCE WRAPPER
# ============================================================
def run_yolo(model, img, model_name="YOLO"):
    try:
        print(f"[DEBUG] Running prediction on {model_name}...")

        start = time.time()
        results = model.predict(img)
        print(f"[DEBUG] Prediction time: {time.time() - start:.4f} seconds")

        if not results:
            return "-"

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("[WARN] No detections found.")
            return "-"

        best = result.boxes[0]
        cls_id = int(best.cls[0])
        conf = float(best.conf[0])
        cls_name = result.names[cls_id]

        print(f"[DEBUG] TOP PREDICTION -> {cls_name} ({conf:.4f})")

        return cls_name if conf > 0.5 else "-"

    except Exception as e:
        print(f"[ERROR] YOLO prediction error: {e}")
        raise


# ============================================================
# ENDPOINT: PREDIKSI HURUF
# ============================================================
@app.post("/predict")
async def predict_letter(request: ImageRequest):
    print("\n===== [REQUEST /predict - LETTER] =====")
    try:
        if letter_model is None:
            return {"error": "Model YOLO huruf tidak dimuat"}

        img = decode_base64_image(request.image)
        pred = run_yolo(letter_model, img, "YOLO-HURUF")

        return {"prediction": pred}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ENDPOINT: PREDIKSI KATA
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(request: ImageRequest):
    print("\n===== [REQUEST /predict-sentence] =====")
    try:
        if sentence_model is None:
            return {"error": "Model YOLO kata tidak dimuat"}

        img = decode_base64_image(request.image)
        pred = run_yolo(sentence_model, img, "YOLO-KATA")

        return {"prediction": pred}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# ENDPOINT: PREDIKSI OBJEK
# ============================================================
@app.post("/predict-object")
async def predict_object(request: ImageRequest):
    print("\n===== [REQUEST /predict-object] =====")
    try:
        if object_model is None:
            return {"error": "Model YOLO object tidak dimuat"}

        img = decode_base64_image(request.image)
        pred = run_yolo(object_model, img, "YOLO-OBJECT")

        return {"prediction": pred}

    except Exception as e:
        return {"error": str(e)}
