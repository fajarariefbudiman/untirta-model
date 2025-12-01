from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io
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
# LOAD MODEL YOLO HURUF
# ============================================================
letter_model_path = "best.pt"
try:
    letter_model = YOLO(letter_model_path)
    print(f"Model YOLO HURUF '{letter_model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model YOLO HURUF: {e}")
    letter_model = None

# ============================================================
# LOAD MODEL YOLO KATA
# ============================================================
sentence_model_path = "best_sentence.pt"
try:
    sentence_model = YOLO(sentence_model_path)
    print(f"Model YOLO KATA '{sentence_model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model YOLO KATA: {e}")
    sentence_model = None


# ============================================================
# Base64 Request Model
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"message": "Server SIBI Aktif (Huruf + Kata)"}


# ============================================================
# FUNGSI PEMBANTU (DECODE BASE64)
# ============================================================
def decode_base64_image(b64_string):
    image_data = b64_string.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(image_bytes))
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return img


# ============================================================
# 1. ENDPOINT PREDIKSI HURUF
# ============================================================
@app.post("/predict")
async def predict_letter(request: ImageRequest):
    if letter_model is None:
        return {"error": "Model YOLO huruf tidak dimuat"}

    try:
        img = decode_base64_image(request.image)

        results = letter_model.predict(img)
        prediction_text = "-"

        if results and results[0].boxes:
            best_box = results[0].boxes[0]
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            cls_name = results[0].names[cls_id]

            if conf > 0.5:
                prediction_text = cls_name

        return {"prediction": prediction_text}

    except Exception as e:
        print("Error prediksi huruf:", e)
        return {"error": str(e)}


# ============================================================
# 2. ENDPOINT PREDIKSI KATA / KALIMAT
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(request: ImageRequest):
    if sentence_model is None:
        return {"error": "Model YOLO kata tidak dimuat"}

    try:
        img = decode_base64_image(request.image)

        results = sentence_model.predict(img)
        prediction_text = "-"

        if results and results[0].boxes:
            best_box = results[0].boxes[0]
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            cls_name = results[0].names[cls_id]

            if conf > 0.5:
                prediction_text = cls_name

        return {"prediction": prediction_text}

    except Exception as e:
        print("Error prediksi kata:", e)
        return {"error": str(e)}
