from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io
from PIL import Image
from tensorflow.keras.models import load_model

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
# 1. LOAD YOLO MODEL (HURUF A-Z)
# ============================================================
model_letter_path = "best.pt"
try:
    model_letter = YOLO(model_letter_path)
    print(f"[OK] Model huruf '{model_letter_path}' dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal load model huruf: {e}")
    model_letter = None

# ============================================================
# 2. LOAD KERAS MODEL (KATA)
# ============================================================
keras_model_path = "my_model.keras"
try:
    keras_model = load_model(keras_model_path)
    print(f"[OK] Model kata '{keras_model_path}' dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal load model Keras: {e}")
    keras_model = None

label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}

# ============================================================
# 3. LOAD YOLO MODEL LINGKUNGAN (mauri.pt)
# ============================================================
model_mauri_path = "mauri.pt"
try:
    model_mauri = YOLO(model_mauri_path)
    print(f"[OK] Model Mauri '{model_mauri_path}' dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal load model Mauri: {e}")
    model_mauri = None

# ============================================================
# BASE64 Request Format
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"message": "Server SIBI Aktif (Huruf + Kata + Mauri Object Detection)"}


# ============================================================
# 4. ENDPOINT YOLO HURUF (best.pt)
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    if model_letter is None:
        return {"error": "Model huruf tidak berhasil dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        results = model_letter.predict(img)
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
        return {"error": str(e)}


# ============================================================
# 5. ENDPOINT KERAS KATA (my_model.keras)
# ============================================================
def preprocess_for_keras(pil_image, size=(228, 228)):
    img = pil_image.resize(size)
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    if keras_model is None:
        return {"error": "Model kata tidak berhasil dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed = preprocess_for_keras(pil_image)

        predictions = keras_model.predict(processed)
        predicted_index = int(np.argmax(predictions))
        predicted_word = label_map.get(predicted_index, "-")

        return {"prediction": predicted_word}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 6. ENDPOINT OBJECT DETECTION LINGKUNGAN (mauri.pt)
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    if model_mauri is None:
        return {"error": "Model Mauri tidak berhasil dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        results = model_mauri.predict(img)

        objects = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls_id]

                objects.append({
                    "object": name,
                    "confidence": round(conf, 3)
                })

        return {"detected_objects": objects}

    except Exception as e:
        return {"error": str(e)}
