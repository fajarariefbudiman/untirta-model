from ultralytics import YOLO

# Load model
model = YOLO("best_object.pt")

# Print labels
print("\n=== LABELS (model.names) ===")
for k, v in model.names.items():
    print(k, ":", v)
