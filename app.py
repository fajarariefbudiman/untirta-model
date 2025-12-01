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
# 1. LOAD YOLO MODEL (FAST MODE)
# ============================================================
def load_yolo(path):
    try:
        model = YOLO(path)
        model.fuse()  # percepat CPU inference
        print(f"[OK] YOLO '{path}' dimuat (FAST mode).")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load {path}: {e}")
        return None


model_letter = load_yolo("best.pt")
model_mauri = load_yolo("mauri.pt")

# ============================================================
# 2. LOAD KERAS MODEL
# ============================================================
try:
    keras_model = load_model("my_model.keras")
    keras_model.compile()
    print("[OK] Model Keras dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal load model Keras: {e}")
    keras_model = None

label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}


# ============================================================
# BASE64 Request Format
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"message": "Server AI aktif (Optimized)"}


# ============================================================
# FAST YOLO LETTER PREDICT
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    try:
        if model_letter is None:
            return {"error": "Model tidak siap"}

        image_data = item.image.split(",")[1]
        img = np.array(Image.open(io.BytesIO(base64.b64decode(image_data))))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_letter(img, imgsz=320, conf=0.5, device="cpu", verbose=False)

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            return {"prediction": results[0].names[cls_id]}

        return {"prediction": "-"}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SENTENCE PREDICT
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    try:
        if keras_model is None:
            return {"error": "Model tidak siap"}

        img = Image.open(io.BytesIO(base64.b64decode(item.image.split(",")[1])))
        img = img.resize((228, 228))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, 0)

        pred = keras_model.predict(img, verbose=0)
        idx = int(np.argmax(pred))
        return {"prediction": label_map.get(idx, "-")}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MAURI OBJECT DETECTION (FAST MODE)
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    try:
        if model_mauri is None:
            return {"error": "Model tidak siap"}

        image_data = item.image.split(",")[1]
        img = np.array(Image.open(io.BytesIO(base64.b64decode(image_data))))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_mauri(img, imgsz=320, device="cpu", verbose=False)

        objects = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                objects.append(
                    {
                        "object": results[0].names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 3),
                    }
                )

        return {"detected_objects": objects}

    except Exception as e:
        return {"error": str(e)}


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
# 1. LOAD YOLO MODEL (FAST MODE)
# ============================================================
def load_yolo(path):
    try:
        model = YOLO(path)
        model.fuse()  # percepat CPU inference
        print(f"[OK] YOLO '{path}' dimuat (FAST mode).")
        return model
    except Exception as e:
        print(f"[ERROR] Gagal load {path}: {e}")
        return None


model_letter = load_yolo("best.pt")
model_mauri = load_yolo("mauri.pt")

# ============================================================
# 2. LOAD KERAS MODEL
# ============================================================
try:
    keras_model = load_model("my_model.keras")
    keras_model.compile()
    print("[OK] Model Keras dimuat.")
except Exception as e:
    print(f"[ERROR] Gagal load model Keras: {e}")
    keras_model = None

label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}


# ============================================================
# BASE64 Request Format
# ============================================================
class ImageRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"message": "Server AI aktif (Optimized)"}


# ============================================================
# FAST YOLO LETTER PREDICT
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    try:
        if model_letter is None:
            return {"error": "Model tidak siap"}

        image_data = item.image.split(",")[1]
        img = np.array(Image.open(io.BytesIO(base64.b64decode(image_data))))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_letter(img, imgsz=320, conf=0.5, device="cpu", verbose=False)

        if results and results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls[0])
            return {"prediction": results[0].names[cls_id]}

        return {"prediction": "-"}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# SENTENCE PREDICT
# ============================================================
@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    try:
        if keras_model is None:
            return {"error": "Model tidak siap"}

        img = Image.open(io.BytesIO(base64.b64decode(item.image.split(",")[1])))
        img = img.resize((228, 228))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, 0)

        pred = keras_model.predict(img, verbose=0)
        idx = int(np.argmax(pred))
        return {"prediction": label_map.get(idx, "-")}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# MAURI OBJECT DETECTION (FAST MODE)
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    try:
        if model_mauri is None:
            return {"error": "Model tidak siap"}

        image_data = item.image.split(",")[1]
        img = np.array(Image.open(io.BytesIO(base64.b64decode(image_data))))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (320, 320))

        results = model_mauri(img, imgsz=320, device="cpu", verbose=False)

        objects = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                objects.append(
                    {
                        "object": results[0].names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 3),
                    }
                )

        return {"detected_objects": objects}

    except Exception as e:
        return {"error": str(e)}
