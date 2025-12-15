import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr
import os
import time
import re

# Set page config
st.set_page_config(page_title="Number Plate Detection & OCR", page_icon="🚗", layout="wide")

# Session State for Resetting
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def reset_app():
    st.session_state.upload_key += 1
    # Clear any other session state variables if needed
    if 'ocr_results' in st.session_state:
        del st.session_state['ocr_results']

# Custom CSS for premium Dark Mode UI
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .main {
        background-color: #000000;
    }
    h1 {
        background: linear-gradient(to right, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .stCard {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(255, 255, 255, 0.1), 0 2px 4px -1px rgba(255, 255, 255, 0.06);
        border: 1px solid #333;
    }
    .result-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fbbf24; /* Amber-400 for high visibility */
        font-family: 'Courier New', monospace;
        text-align: center;
        letter-spacing: 0.1em;
        background-color: #000;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #fbbf24;
    }
    div[data-testid="stFileUploader"] {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Intelligent Number Plate Recognition")

# Initialize OCR
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False) # Set gpu=True if CUDA is available

reader = get_ocr_reader()

# Load Model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Try to find the best model path automatically
possible_paths = [
    r"yolov8_number_plate/weights/best.pt",
    os.path.join("runs", "detect", "yolov8_number_plate", "weights", "best.pt"),
    os.path.join("runs", "detect", "train", "weights", "best.pt"),
    "yolov8n.pt" 
]

model_path = None
for p in possible_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path:
    # Sidebar
    st.sidebar.title("🛠️ Configuration")
    st.sidebar.success(f"Model: {os.path.basename(model_path)}")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.01)
    ocr_threshold = st.sidebar.slider("OCR Confidence", 0.0, 1.0, 0.20, 0.01)
    
    # Add Clear Button to sidebar
    if st.sidebar.button("🧹 Clear / Reset", on_click=reset_app):
        pass # The callback handles the reset logic

    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
else:
    st.error("No model found. Please train the model first.")
    st.stop()

uploaded_file = st.file_uploader("Drop your vehicle image here...", type=["jpg", "jpeg", "png"], key=f"uploader_{st.session_state.upload_key}")

if uploaded_file is not None:
    # Display original image
    # Reset file pointer to be safe for re-runs
    uploaded_file.seek(0)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("📸 Original Input")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Perform inference
    with st.spinner('Scanning number plate...'):
        start_time = time.time()
        results = model.predict(img_array, conf=confidence_threshold)
        
        annotated_img = img_array.copy()
        
        # Use a list to collect valid plates
        valid_plates_found = []
        
        # Debug Info Collection
        debug_logs = []
        debug_logs.append(f"Model detected {len(results[0].boxes)} bounding boxes.")

        for i, result in enumerate(results):
            boxes = result.boxes
            for j, box in enumerate(boxes):
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                debug_logs.append(f"Box {j+1}: Conf={conf:.2f}, Coords=({x1},{y1})-({x2},{y2})")

                # Crop the number plate for text OCR
                plate_crop = img_array[y1:y2, x1:x2]
                
                # Helper function to rotate image
                def rotate_image(image, angle):
                    image_center = tuple(np.array(image.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
                    return result

                # OCR Retry Logic for Rotated Images
                best_text = ""
                best_conf = 0.0
                
                angles_to_try = [0, -10, 10, -15, 15, -5, 5]
                log_entry = f"Box {j+1} OCR: "
                
                for angle in angles_to_try:
                    # Rotate if angle is not 0
                    if angle == 0:
                        processed_crop = plate_crop.copy()
                    else:
                        processed_crop = rotate_image(plate_crop, angle)

                    if processed_crop.size > 0:
                        try:
                            # Preprocessing for OCR
                            gray_plate = cv2.cvtColor(processed_crop, cv2.COLOR_RGB2GRAY)
                            
                            # Upscale if too small
                            if gray_plate.shape[0] < 50 or gray_plate.shape[1] < 150:
                                scale = 3.0
                                gray_plate = cv2.resize(gray_plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                            
                            # Contrast enhancement
                            gray_plate = cv2.equalizeHist(gray_plate)
                            
                            # Denoise
                            gray_plate = cv2.fastNlMeansDenoising(gray_plate, None, 10, 7, 21)

                            text_detections = reader.readtext(gray_plate)
                            
                            detected_text = ""
                            local_conf = 0.0
                            count = 0
                            
                            for detection in text_detections:
                                bbox, text, score = detection
                                if score > ocr_threshold:
                                    detected_text += text + " "
                                    local_conf += score
                                    count += 1
                            
                            if count > 0:
                                local_conf /= count
                            
                            detected_text = detected_text.strip()
                            clean_text = re.sub(r'[^A-Z0-9]', '', detected_text.upper())
                            
                            if clean_text:
                                log_entry += f"[{angle}°: '{clean_text}' ({local_conf:.2f})] "

                            # Check if this result is better
                            if len(clean_text) >= 4:
                                # Prioritize pattern matches
                                is_pattern_match = bool(re.search(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', clean_text))
                                
                                # Boost confidence for pattern matches
                                effective_conf = local_conf + (0.5 if is_pattern_match else 0.0)
                                
                                if effective_conf > best_conf:
                                    best_conf = effective_conf
                                    best_text = clean_text
                                    
                        except Exception as e:
                            print(f"OCR Error at angle {angle}: {e}")
                
                debug_logs.append(log_entry)

                # Process the best result found across all angles
                if best_text:
                    valid_plates_found.append(best_text)
                    
                    # Draw clean text on image
                    t_size = cv2.getTextSize(best_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    # Ensure text doesn't go off-screen top
                    text_y = y1 - 10 if y1 - 10 > 25 else y1 + 25 
                    
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.rectangle(annotated_img, (x1, text_y - 25), (x1 + t_size[0] + 10, text_y + 5), (0, 0, 0), -1)
                    cv2.putText(annotated_img, best_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


        end_time = time.time()
        
    with col2:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("🎯 Detection Result")
        st.image(annotated_img, use_column_width=True)
        st.caption(f"Processed in {end_time - start_time:.2f} seconds")
        st.markdown("</div>", unsafe_allow_html=True)

    # FINAL RESULT DISPLAY
    st.markdown("---")
    
    # Robust Output Logic
    if valid_plates_found:
        st.markdown("<h2 style='text-align: center;'>Detected Number Plate</h2>", unsafe_allow_html=True)
        # Use set to remove duplicates, then list to iterate
        unique_plates = list(set(valid_plates_found))
        for plate_text in unique_plates:
            st.markdown(f"<div class='result-text'>{plate_text}</div>", unsafe_allow_html=True)
    else:
        # Check why found nothing
        if len(results[0].boxes) > 0:
            st.warning("⚠️ Vehicle/Number Plate detected, but OCR could not read the text clearly.")
            st.markdown("**Troubleshooting:**\n- Try adjusting the **OCR Confidence** slider in the sidebar.\n- Ensure the image is not too blurry.\n- Check Debug Logs below.")
        else:
            st.error("❌ No number plate detected in this image.")
            st.markdown("**Troubleshooting:**\n- Try adjusting the **Detection Confidence** slider in the sidebar (lower it).\n- Ensure the image contains a visible vehicle.")

    # Debug Expander
    with st.expander("🔍 Debug Logs (Click to expand)"):
        st.write("System Logs:")
        for log in debug_logs:
            st.text(log)



