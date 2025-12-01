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

# === CORS (WAJIB untuk front-end) ===
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
model_path = "best.pt"
try:
    model = YOLO(model_path)
    print(f"Model YOLO '{model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model YOLO: {e}")
    model = None

# ============================================================
# 2. LOAD KERAS MODEL (KATA)
# ============================================================
keras_model_path = "my_model.keras"
try:
    keras_model = load_model(keras_model_path)
    print(f"Model Keras '{keras_model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model Keras: {e}")
    keras_model = None

# ============================================================
# 3. LOAD YOLO MODEL MAURI (OBJECT DETECTION)
# ============================================================
mauri_model_path = "mauri.pt"
try:
    model_mauri = YOLO(mauri_model_path)
    print(f"Model YOLO MAURI '{mauri_model_path}' berhasil dimuat.")
except Exception as e:
    print(f"GAGAL memuat model MAURI: {e}")
    model_mauri = None

# Label kata sesuai output model
label_map = {0: "saya", 1: "mau", 2: "beli", 3: "terima_kasih", 4: "tolong"}

# Base64 Request Model
class ImageRequest(BaseModel):
    image: str

@app.get("/")
def home():
    return {"message": "Server SIBI Aktif (Huruf + Kata + MAURI)"}

# ============================================================
# 4. ENDPOINT YOLO (HURUF)
# ============================================================
@app.post("/predict")
async def predict_letter(item: ImageRequest):
    if model is None:
        return {"error": "Model YOLO tidak berhasil dimuat"}
    
    try:
        # Decode Base64
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Prediksi YOLO
        results = model.predict(img)
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
        print(f"Error prediksi YOLO: {e}")
        return {"error": str(e)}

# ============================================================
# 5. ENDPOINT KERAS (KATA)
# ============================================================
def preprocess_for_keras(pil_image, size=(228, 228)):  
    img = pil_image.resize(size)
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict-sentence")
async def predict_sentence(item: ImageRequest):
    if keras_model is None:
        return {"error": "Model Keras tidak berhasil dimuat"}
    
    try:
        # Decode Base64
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess
        processed = preprocess_for_keras(pil_image)
        
        # Prediksi
        predictions = keras_model.predict(processed)
        predicted_index = int(np.argmax(predictions))
        predicted_word = label_map.get(predicted_index, "-")
        
        return {"prediction": predicted_word}
    
    except Exception as e:
        print(f"Error prediksi Keras: {e}")
        return {"error": str(e)}

# ============================================================
# 6. ENDPOINT MAURI (OBJECT DETECTION)
# ============================================================
@app.post("/predict-mauri")
async def predict_mauri(item: ImageRequest):
    if model_mauri is None:
        return {"error": "Model MAURI tidak berhasil dimuat"}
    
    try:
        # Decode Base64
        image_data = item.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Prediksi MAURI dengan confidence threshold lebih rendah
        results = model_mauri.predict(img, conf=0.25, verbose=False)
        
        detected_objects = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = results[0].names[cls_id]
                
                detected_objects.append({
                    "object": cls_name,
                    "confidence": round(conf, 3)
                })
        
        return {"detected_objects": detected_objects}
    
    except Exception as e:
        print(f"Error prediksi MAURI: {e}")
        return {"error": str(e)}