import requests
import base64
import sys
import os

# ===============================
# Konfigurasi URL server
# ===============================
BASE_URL = "http://103.49.239.237:8000"
YOLO_URL = f"{BASE_URL}/predict"
KERAS_URL = f"{BASE_URL}/predict-sentence"
MAURI_URL = f"{BASE_URL}/predict-mauri"

# ===============================
# Fungsi untuk encode gambar
# ===============================
def encode_image(file_path):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

# ===============================
# Test YOLO huruf
# ===============================
def test_yolo(image_path):
    payload = {"image": encode_image(image_path)}
    try:
        resp = requests.post(YOLO_URL, json=payload, timeout=10)
        print(f"[YOLO] {image_path} -> {resp.json()}")
    except Exception as e:
        print(f"[YOLO ERROR] {image_path} -> {e}")

# ===============================
# Test Keras kalimat
# ===============================
def test_keras(image_path):
    payload = {"image": encode_image(image_path)}
    try:
        resp = requests.post(KERAS_URL, json=payload, timeout=10)
        print(f"[Keras] {image_path} -> {resp.json()}")
    except Exception as e:
        print(f"[Keras ERROR] {image_path} -> {e}")

# ===============================
# Test MAURI object detection
# ===============================
def test_mauri(image_path):
    payload = {"image": encode_image(image_path)}
    try:
        resp = requests.post(MAURI_URL, json=payload, timeout=10)
        print(f"[MAURI] {image_path} -> {resp.json()}")
    except Exception as e:
        print(f"[MAURI ERROR] {image_path} -> {e}")

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_gambar1> [<path_gambar2> ...]")
        sys.exit(1)

    for img_path in sys.argv[1:]:
        if not os.path.exists(img_path):
            print(f"File tidak ditemukan: {img_path}")
            continue

        test_yolo(img_path)
        test_keras(img_path)
        test_mauri(img_path)
