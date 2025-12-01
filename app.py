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
# LOAD YOLO HURUF
# ============================================================
letter_model_path = "best.pt"
try:
    print("[INFO] Loading YOLO letter model...")
    letter_model = YOLO(letter_model_path)
    print(f"[SUCCESS] YOLO Letter Model loaded: {letter_model_path}")
except Exception as e:
    print(f"[ERROR] Failed loading YOLO Letter Model: {e}")
    letter_model = None

# ============================================================
# LOAD YOLO KATA
# ============================================================
sentence_model_path = "best_sentence.pt"
try:
    print("[INFO] Loading YOLO sentence model...")
    sentence_model = YOLO(sentence_model_path)
    print(f"[SUCCESS] YOLO Sentence Model loaded: {sentence_model_path}")
except Exception as e:
    print(f"[ERROR] Failed loading YOLO Sentence Model: {e}")
    sentence_model = None


# ============================================================
# DATA MODEL
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    print("[DEBUG] Root endpoint accessed")
    return {"message": "Server SIBI Aktif (Huruf + Kata)"}


# ============================================================
# HELPER: Decode Base64 → OpenCV Image (with debugging)
# ============================================================
def decode_base64_image(b64_string):
    try:
        print("[DEBUG] Starting Base64 decode...")

        if "," in b64_string:
            image_data = b64_string.split(",")[1]
        else:
            image_data = b64_string

        # Decode
        image_bytes = base64.b64decode(image_data)
        print(f"[DEBUG] Base64 decoded size: {len(image_bytes)} bytes")

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        print(f"[DEBUG] Image converted to OpenCV format: shape={img.shape}")

        return img

    except Exception as e:
        print(f"[ERROR] Failed decoding Base64: {e}")
        raise


# ============================================================
# COMMON YOLO PREDICT FUNCTION (with debugging)
# ============================================================
def run_yolo(model, img, model_name="YOLO"):
    try:
        print(f"[DEBUG] Running prediction on {model_name}...")

        start_time = time.time()
        results = model.predict(img)
        end_time = time.time()

        print(f"[DEBUG] Prediction time: {end_time - start_time:.4f} seconds")

        if not results:
            print("[WARN] No result returned from YOLO")
            return "-"

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("[WARN] YOLO detected 0 objects")
            return "-"

        print(f"[DEBUG] YOLO detected {len(result.boxes)} boxes")

        # Ambil box pertama (paling confident)
        best_box = result.boxes[0]
        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        cls_name = result.names[cls_id]

        print(f"[DEBUG] TOP PREDICTION → Class: {cls_name} | ID: {cls_id} | Conf: {conf:.4f}")

        if conf > 0.5:
            print("[DEBUG] Confidence OK, sending result back")
            return cls_name
        else:
            print("[DEBUG] Confidence too low")
            return "-"

    except Exception as e:
        print(f"[ERROR] YOLO prediction error: {e}")
        raise


# ============================================================
# ENDPOINT: PREDIKSI HURUF
# ============================================================
@app.post("/predict")
async def predict_letter(request: ImageRequest):
    print("\n================= [REQUEST /predict] =================")
    try:
        if letter_model is None:
            print("[ERROR] Letter model is NOT loaded")
            return {"error": "Model YOLO huruf tidak dimuat"}

        print("[DEBUG] Received image request")
        img = decode_base64_image(request.image)

        prediction = run_yolo(letter_model, img, model_name="YOLO-HURUF")

        print(f"[DONE] Final Letter Prediction: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        print(f"[FATAL ERROR] /predict failed: {e}")
        return {"error": str(e)}


# ============================================================
# ENDPOINT: PREDIKSI KATA
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(request: ImageRequest):
    print("\n================= [REQUEST /predict-sentence] =================")
    try:
        if sentence_model is None:
            print("[ERROR] Sentence model is NOT loaded")
            return {"error": "Model YOLO kata tidak dimuat"}

        print("[DEBUG] Received image request")
        img = decode_base64_image(request.image)

        prediction = run_yolo(sentence_model, img, model_name="YOLO-KATA")

        print(f"[DONE] Final Sentence Prediction: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        print(f"[FATAL ERROR] /predict-sentence failed: {e}")
        return {"error": str(e)}
