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

# === CORS (WAJIB) ===
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
    model_letter.fuse()
    print(f"✓ Model Huruf berhasil dimuat.")
except Exception as e:
    print(f"✗ Gagal load model huruf: {e}")
    model_letter = None

# ============================================================
# 2. LOAD YOLO MODEL (KALIMAT)
# ============================================================
model_sentence_path = "best_sentence.pt"

try:
    model_sentence = YOLO(model_sentence_path)
    model_sentence.fuse()
    print("✓ Model Sentence berhasil dimuat.")
except Exception as e:
    print(f"✗ Gagal load model sentence: {e}")
    model_sentence = None

# ============================================================
# 3. LOAD YOLO MAURI
# ============================================================
mauri_path = "mauri.pt"

try:
    model_mauri = YOLO(mauri_path)
    model_mauri.fuse()
    print("✓ Model MAURI berhasil dimuat.")
except Exception as e:
    print(f"✗ Gagal load model MAURI: {e}")
    model_mauri = None


# Label kalimat
label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}


# Base64 Request
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"message": "Server SIBI aktif (Huruf + Kalimat + Mauri)"}


# ============================================================
# 4. ENDPOINT HURUF
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    if model_letter is None:
        return {"error": "Model huruf tidak dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_letter.predict(
            img,
            imgsz=320,
            conf=0.5,
            device="cpu",
            verbose=False
        )

        prediction_text = "-"

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_id]

            if conf > 0.5:
                prediction_text = cls_name

        return {"prediction": prediction_text}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 5. ENDPOINT KALIMAT (YOLO)
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    if model_sentence is None:
        return {"error": "Model kalimat tidak dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_sentence.predict(
            img,
            imgsz=320,
            conf=0.5,
            device="cpu",
            verbose=False
        )

        prediction = "-"

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            prediction = label_map.get(cls_id, "-")

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 6. ENDPOINT MAURI
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    if model_mauri is None:
        return {"error": "Model MAURI tidak dimuat"}

    try:
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_mauri.predict(
            img,
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        detected = []

        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]

                detected.append({
                    "object": cls_name,
                    "confidence": round(conf, 3)
                })

        return {"detected_objects": detected}

    except Exception as e:
        return {"error": str(e)}
