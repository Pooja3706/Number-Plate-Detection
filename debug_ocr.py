import cv2
import easyocr
from ultralytics import YOLO
import os
import re

# Paths
img_path = r"C:/Users/POOJA/.gemini/antigravity/brain/defda37c-c8a4-453e-bd92-9acf52c8fede/uploaded_image_1765619292082.jpg"

# Load Model
possible_paths = [
    r"yolov8_number_plate/weights/best.pt",
    os.path.join("runs", "detect", "yolov8_number_plate", "weights", "best.pt"),
    os.path.join("runs", "detect", "train", "weights", "best.pt"),
    "yolov8n.pt" 
]
model_path = "yolov8n.pt"
for p in possible_paths:
    if os.path.exists(p):
        model_path = p
        break

print(f"Loading model: {model_path}")
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=False)

# Inference
print(f"Processing image: {img_path}")
results = model.predict(img_path, conf=0.25)

img = cv2.imread(img_path)

print("\n--- Raw Detections ---")
for result in results:
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        print(f"Box {i+1}: Conf={conf:.2f}, Coords={x1},{y1},{x2},{y2}")
        
        # Crop
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        # Preprocess
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # OCR
        detections = reader.readtext(gray)
        print(f"  OCR Raw Output:")
        for bbox, text, score in detections:
            print(f"    - '{text}' (score: {score:.2f})")
            
            # Simulated Filter Check
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            print(f"      -> Cleaned: '{clean_text}'")
            
            # Indian Plate Regex Check
            # Pattern: 2 Char State, 1-2 Digit District, 0-3 Char Series, 4 Digit Number
            pattern = r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$'
            match = re.search(pattern, clean_text)
            if match:
                print(f"      [MATCH] Possible Plate: {clean_text}")
            else:
                print(f"      [x] Ignored (No pattern match)")

