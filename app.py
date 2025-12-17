import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import easyocr
import os
import time
import re
import pandas as pd
from datetime import datetime

# --- PAGE CONFIGURAION ---
st.set_page_config(
    page_title="NeuroPlate | Intelligent Recognition", 
    page_icon="🚘", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AESTHETIC STYLING (GLASSMORPHISM) ---
def load_css():
    st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #172554 100%);
            color: #e2e8f0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        
        /* Glassmorphic Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
            margin-bottom: 24px;
            transition: transform 0.2s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* Result Display Typography */
        .plate-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 3rem;
            font-weight: 700;
            color: #fbbf24;
            text-shadow: 0 0 20px rgba(251, 191, 36, 0.3);
            text-align: center;
            background: rgba(0, 0, 0, 0.5);
            padding: 1rem 2rem;
            border-radius: 12px;
            border: 1px solid #fbbf24;
            margin: 1.5rem 0;
            letter-spacing: 4px;
        }
        
        .state-badge {
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1rem;
            color: white;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }

        /* Custom Button Styling */
        div.stButton > button {
            background: linear-gradient(92deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.6);
            transform: translateY(-1px);
        }

        /* Sidebar Beautification */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- STATE MAPPING ---
STATE_MAP = {
    "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar",
    "CG": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana",
    "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
    "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TS": "Telangana",
    "TR": "Tripura", "UP": "Uttar Pradesh", "UK": "Uttarakhand", "WB": "West Bengal",
    "AN": "Andaman & Nicobar", "CH": "Chandigarh", "DN": "Dadra & Nagar Haveli",
    "DD": "Daman & Diu", "DL": "Delhi", "JK": "Jammu & Kashmir", "LA": "Ladakh",
    "LD": "Lakshadweep", "PY": "Puducherry"
}

def detect_state(plate_text):
    if not plate_text: return "Unknown"
    code = plate_text[:2].upper()
    return STATE_MAP.get(code, "Unknown Region")

# --- SESSION STATE ---
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

def add_to_history(filename, plate_text, conf, state_name):
    timestamp = datetime.now().strftime("%I:%M %p | %d %b")
    new_entry = {
        "ID": len(st.session_state.results_history) + 1,
        "Timestamp": timestamp,
        "Filename": filename,
        "Detected Plate": plate_text,
        "State": state_name,
        "Confidence": f"{conf:.2f}",
        "Verified": False # Allow user to mark as verified
    }
    st.session_state.results_history.insert(0, new_entry)

# --- MODEL LOADING ---
@st.cache_resource
def get_resources():
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Locate YOLO Model
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
            
    model = YOLO(model_path) if model_path else None
    return reader, model, model_path

# --- PAGES ---

def page_dashboard(reader, model):
    st.title("🚘 NeuroPlate Logic")
    st.markdown("### Vehicle Identity & Region Scanner")
    st.write("") # Spacer

    uploaded_file = st.file_uploader("Drop image here to analyze...", type=["jpg", "png", "jpeg"], key=f"uploader_{st.session_state.upload_key}")

    if uploaded_file:
        image = Image.open(uploaded_file)
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image) # Fix phone rotation
        except: pass
        img_array = np.array(image)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.caption("SOURCE FEED")
            st.image(image, use_column_width=True, channels="RGB")
            st.markdown("</div>", unsafe_allow_html=True)

        # Process on Load or Button? Let's do auto-process if file is there for better UX
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.caption("ANALYSIS FEED")
            
            with st.spinner("Decoding Neural Patterns..."):
                start_time = time.time()
                
                # Inference
                results = model.predict(img_array, conf=0.25)
                annotated_img = img_array.copy()
                
                best_text = None
                best_conf = 0.0

                if results[0].boxes:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Draw aesthetic box
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 200), 2)
                        
                        # OCR
                        crop = img_array[y1:y2, x1:x2]
                        try:
                            # Preprocess
                            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                            # Upscale if small
                            if gray.shape[0] < 50:
                                gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                            
                            detections = reader.readtext(gray)
                            
                            # Scoring logic
                            text_accum = ""
                            score_sum = 0
                            cnt = 0
                            for _, txt, score in detections:
                                if score > 0.2:
                                    text_accum += txt + " "
                                    score_sum += score
                                    cnt += 1
                                    
                            if cnt > 0:
                                clean = re.sub(r'[^A-Z0-9]', '', text_accum.upper())
                                final_score = score_sum / cnt
                                # Pattern boost
                                if re.search(r'[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}', clean):
                                    final_score += 0.4 
                                
                                if final_score > best_conf:
                                    best_conf = final_score
                                    best_text = clean
                        except: pass
                
                # Show Annotated Image
                st.image(annotated_img, use_column_width=True, channels="RGB")
                
                if best_text:
                    state = detect_state(best_text)
                    
                    # Aesthetic Result Display
                    st.markdown(f"<div class='plate-number'>{best_text}</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;">
                            <div class="state-badge">🇮🇳 {state}</div>
                            <div class="state-badge" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">Conf: {int(best_conf*100)}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Auto-save
                    # Prevent double-save: check if most recent entry is same image
                    should_save = True
                    if st.session_state.results_history:
                        last = st.session_state.results_history[0]
                        if last['Filename'] == uploaded_file.name and last['Detected Plate'] == best_text:
                            should_save = False
                    
                    if should_save:
                        add_to_history(uploaded_file.name, best_text, best_conf, state)
                        st.toast("Scan Recorded Successfully", icon="💾")
                        
                else:
                    st.warning("Vehicle detected, but license plate was unreadable.")
            
            st.markdown("</div>", unsafe_allow_html=True)


def page_results():
    st.title("📂 Result Page")
    st.markdown("Database of all scanned vehicles.")
    
    if not st.session_state.results_history:
        st.info("No data available. Go to **Dashboard** to start scanning.")
        return

    # Data Editor
    df = pd.DataFrame(st.session_state.results_history)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scans", len(df))
    with col2:
        top_state = df['State'].mode()[0] if not df.empty else "N/A"
        st.metric("Most Frequent State", top_state)
    
    st.markdown("### Scanned Records")
    
    edited_df = st.data_editor(
        df,
        key="results_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Timestamp": st.column_config.TextColumn("Time", disabled=True),
            "Detected Plate": st.column_config.TextColumn("Plate Number", required=True),
            "State": st.column_config.SelectboxColumn("State", options=list(STATE_MAP.values())),
            "Verified": st.column_config.CheckboxColumn("Verify?", default=False),
        },
        hide_index=True
    )
    
    st.markdown("---")
    if st.button("📥 Download CSV Report"):
        csv = edited_df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=f"neuroplate_report_{int(time.time())}.csv",
            mime="text/csv"
        )


def main():
    try:
        reader, model, path = get_resources()
    except Exception as e:
        st.error(f"System Error: {e}")
        st.stop()
        
    if not model:
        st.error("Model Weights Not Found. Please re-train.")
        st.stop()

    # Aesthetic Sidebar
    st.sidebar.markdown("## ⚡ Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Result Page"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Using Engine: `{os.path.basename(path)}`")
    
    if page == "Dashboard":
        page_dashboard(reader, model)
    elif page == "Result Page":
        page_results()

if __name__ == "__main__":
    main()
