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
# LOAD MODEL
# ============================================================
def load_model(path, nama):
    try:
        model = YOLO(path)
        model.fuse()
        print(f"✓ Model {nama} berhasil dimuat: {path}")
        return model
    except Exception as e:
        print(f"✗ Gagal load model {nama}: {e}")
        return None


model_letter = load_model("best.pt", "Huruf")
model_sentence = load_model("best_sentence.pt", "Kalimat")
model_mauri = load_model("mauri.pt", "MAURI")

label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}


class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    print("Endpoint / diakses.")
    return {"message": "Server SIBI aktif (Huruf + Kalimat + Mauri)"}


# ============================================================
# PREDICT HURUF (INPUT 320×320)
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    print("\n=== [HIT] /predict (Huruf) ===")

    if model_letter is None:
        print("Model huruf tidak dimuat!")
        return {"error": "Model huruf tidak dimuat"}

    try:
        print("Base64 length:", len(item.image))

        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        print("Gambar diterima:", pil_image.size, pil_image.mode)

        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_letter.predict(
            img, imgsz=320, conf=0.5, device="cpu", verbose=False
        )

        print("Jumlah box terdeteksi:", len(results[0].boxes))

        prediction_text = "-"

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls_id]

            print(f"Class: {cls_name}, Confidence: {conf}")

            if conf > 0.5:
                prediction_text = cls_name

        print("Final Prediction:", prediction_text)
        return {"prediction": prediction_text}

    except Exception as e:
        print("ERROR (huruf):", e)
        return {"error": str(e)}


# ============================================================
# PREDICT KALIMAT (INPUT 640×640)
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    print("\n=== [HIT] /predict-sentence (Kalimat) ===")

    if model_sentence is None:
        print("Model kalimat tidak dimuat!")
        return {"error": "Model kalimat tidak dimuat"}

    try:
        print("Base64 length:", len(item.image))

        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        print("Gambar diterima:", pil_image.size, pil_image.mode)

        # Resize ke 640
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 640))

        results = model_sentence.predict(
            img, imgsz=640, conf=0.5, device="cpu", verbose=False
        )

        print("Jumlah box terdeteksi:", len(results[0].boxes))

        prediction = "-"

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            prediction = label_map.get(cls_id, "-")
            print(f"Class ID: {cls_id}, Sentence: {prediction}")

        print("Final Prediction:", prediction)
        return {"prediction": prediction}

    except Exception as e:
        print("ERROR (kalimat):", e)
        return {"error": str(e)}


# ============================================================
# PREDICT MAURI (INPUT 640×640)
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    print("\n=== [HIT] /predict-mauri (MAURI) ===")

    if model_mauri is None:
        print("Model MAURI tidak dimuat!")
        return {"error": "Model MAURI tidak dimuat"}

    try:
        print("Base64 length:", len(item.image))

        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)

        pil_image = Image.open(io.BytesIO(image_bytes))
        print("Gambar diterima:", pil_image.size, pil_image.mode)

        # Resize ke 640
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (640, 640))

        results = model_mauri.predict(
            img, imgsz=640, conf=0.25, device="cpu", verbose=False
        )

        print("Jumlah box terdeteksi:", len(results[0].boxes))

        detected = []

        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]

                print(f"- OBJ: {cls_name}, CONF: {conf}")

                detected.append({"object": cls_name, "confidence": round(conf, 3)})

        print("Final Detected:", detected)
        return {"detected_objects": detected}

    except Exception as e:
        print("ERROR (mauri):", e)
        return {"error": str(e)}
