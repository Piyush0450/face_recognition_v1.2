"""
FaceRecognition - Stage 11: Streamlit Portfolio Dashboard Entry Point

This interactive web application provides a 4-tab dashboard:
  1. Live Camera: Real-time recognition, live metrics, and automatic attendance logging.
  2. Register New Person: Capture 20 face images and rebuild InsightFace embeddings.
  3. Attendance: Filter daily logs by date, view attendance tables, and export CSV reports.
  4. People: Manage registered individuals, view sample images, and delete profiles.
"""

import datetime
import os
import re
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_threshold
from src.database import (
    delete_person_by_name,
    get_all_embeddings,
    get_all_people,
    get_attendance_by_date,
    get_today_attendance,
    init_db,
    log_attendance,
)

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FaceRecognition Dashboard",
    page_icon="👤",
    layout="wide",
)

# Ensure database is initialized
init_db()


# -----------------------------------------------------------------------------
# Caching & Model Initializers
# -----------------------------------------------------------------------------
@st.cache_resource
def load_insightface_model():
    """Load and cache the InsightFace (buffalo_l) model across app reloads."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    providers = ort.get_available_providers()
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

    app = FaceAnalysis(name="buffalo_l", root=models_dir)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


@st.cache_data(ttl=30)
def load_cached_embeddings():
    """Cache loaded face embeddings from SQLite database to improve performance."""
    return get_all_embeddings()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to a 1D numpy vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def sanitize_name(name: str) -> str:
    """Sanitize input string into a safe folder name."""
    clean_name = name.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_-]", "", clean_name)


# -----------------------------------------------------------------------------
# Sidebar Section
# -----------------------------------------------------------------------------
st.sidebar.title("👤 FaceRecognition")
st.sidebar.markdown("Real-Time Multi-Person Face Recognition & Attendance")
st.sidebar.divider()

# Recognition Threshold Slider
default_threshold = load_threshold()
similarity_threshold = st.sidebar.slider(
    "Recognition Threshold",
    min_value=0.30,
    max_value=0.80,
    value=float(default_threshold),
    step=0.05,
    help="Higher threshold reduces false positives; lower threshold increases matching sensitivity.",
)

st.sidebar.divider()
st.sidebar.markdown("### 📊 System Overview")

all_people = get_all_people()
today_logs = get_today_attendance()

st.sidebar.metric("Registered People", len(all_people))
st.sidebar.metric("Present Today", len(today_logs))
st.sidebar.caption("🤖 **Model Engine:** InsightFace (buffalo_l)")


# -----------------------------------------------------------------------------
# Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Live Camera",
    "➕ Register New Person",
    "📅 Attendance",
    "👥 Registered People",
])


# =============================================================================
# Tab 1: Live Camera
# =============================================================================
with tab1:
    st.header("Live Camera Recognition")
    st.caption("Recognize registered people in real-time and log attendance automatically.")

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        if "camera_running" not in st.session_state:
            st.session_state.camera_running = False

        if not st.session_state.camera_running:
            if st.button("▶️ Start Camera", type="primary"):
                st.session_state.camera_running = True
                st.rerun()
        else:
            if st.button("⏹️ Stop Camera", type="secondary"):
                st.session_state.camera_running = False
                st.rerun()

    metrics_container = st.empty()
    image_container = st.empty()

    if st.session_state.camera_running:
        app = load_insightface_model()
        known_embeddings = load_cached_embeddings()

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Unable to access webcam. Please check camera connection or permissions.")
            st.session_state.camera_running = False
        else:
            prev_time = time.time()

            try:
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        st.error("Failed to grab video frame.")
                        break

                    # Perform face detection
                    detected_faces = app.get(frame)

                    total_faces = len(detected_faces)
                    known_count = 0
                    unknown_count = 0

                    for face in detected_faces:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                        norm_embedding = l2_normalize(face.embedding)

                        best_match = "Unknown"
                        best_score = 0.0

                        for name, known_vec in known_embeddings.items():
                            score = float(np.dot(norm_embedding, known_vec))
                            if score > best_score:
                                best_score = score
                                best_match = name

                        if best_score >= similarity_threshold:
                            color = (0, 255, 0)
                            label = f"{best_match} {int(best_score * 100)}%"
                            known_count += 1
                            # Log attendance automatically in SQLite
                            log_attendance(best_match)
                        else:
                            color = (0, 0, 255)
                            label = "Unknown"
                            unknown_count += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                        cv2.putText(
                            frame,
                            label,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                    # Calculate FPS
                    curr_time = time.time()
                    time_diff = curr_time - prev_time
                    fps = 1.0 / time_diff if time_diff > 0 else 0.0
                    prev_time = curr_time

                    # Update Live Metrics Header
                    with metrics_container.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("FPS", f"{fps:.1f}")
                        m2.metric("Total Faces", total_faces)
                        m3.metric("Known Faces", known_count)
                        m4.metric("Unknown Faces", unknown_count)

                    # Render Frame in Streamlit
                    image_container.image(frame, channels="BGR", use_container_width=True)

            finally:
                cap.release()


# =============================================================================
# Tab 2: Register New Person
# =============================================================================
with tab2:
    st.header("Register New Person")
    st.caption("Capture 20 face images using the webcam, then rebuild face embeddings.")

    col1, col2 = st.columns([2, 1])

    with col1:
        new_name = st.text_input("Person's Name", placeholder="e.g. John Doe").strip()

        if st.button("📷 Capture 20 Images", type="primary"):
            clean_name = sanitize_name(new_name)

            if not clean_name:
                st.warning("Please enter a valid person name.")
            else:
                person_dir = os.path.join(PROJECT_ROOT, "data", "faces", clean_name)
                os.makedirs(person_dir, exist_ok=True)

                cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                face_cascade = cv2.CascadeClassifier(cascade_path)

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Unable to open webcam.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    cap_image_holder = st.empty()

                    captured = 0
                    last_cap_time = 0.0
                    target = 20

                    try:
                        while captured < target:
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                break

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                            if len(faces) == 1:
                                (x, y, w, h) = faces[0]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                curr = time.time()
                                if curr - last_cap_time >= 0.2:
                                    captured += 1
                                    last_cap_time = curr

                                    margin_x = int(w * 0.15)
                                    margin_y = int(h * 0.15)
                                    h_frame, w_frame, _ = frame.shape

                                    x1 = max(0, x - margin_x)
                                    y1 = max(0, y - margin_y)
                                    x2 = min(w_frame, x + w + margin_x)
                                    y2 = min(h_frame, y + h + margin_y)

                                    face_crop = frame[y1:y2, x1:x2]
                                    img_path = os.path.join(person_dir, f"{captured}.jpg")
                                    cv2.imwrite(img_path, face_crop)

                                    progress_bar.progress(captured / target)
                                    status_text.info(f"Captured {captured}/{target} face images for **{clean_name}**.")

                            cap_image_holder.image(frame, channels="BGR", use_container_width=True)

                        if captured >= target:
                            st.success(f"Successfully captured 20 face images for **{clean_name}**!")
                    finally:
                        cap.release()

    with col2:
        st.subheader("Embedding Operations")
        st.caption("After capturing images, generate embeddings and update SQLite database.")

        if st.button("🔄 Rebuild Embeddings", use_container_width=True):
            with st.spinner("Generating embeddings & updating SQLite database..."):
                res1 = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "src", "encode_faces.py")], capture_output=True, text=True)
                res2 = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "src", "migrate_to_db.py")], capture_output=True, text=True)

                if res1.returncode == 0 and res2.returncode == 0:
                    st.cache_data.clear()
                    st.success("Embeddings rebuilt and SQLite database updated successfully!")
                    st.text(res1.stdout)
                else:
                    st.error("Error rebuilding embeddings.")
                    st.code(res1.stderr + "\n" + res2.stderr)


# =============================================================================
# Tab 3: Attendance Reports
# =============================================================================
with tab3:
    st.header("Attendance Records")
    st.caption("View and export daily attendance logs from SQLite database.")

    selected_date = st.date_input("Select Date", datetime.date.today())
    date_str = selected_date.strftime("%Y-%m-%d")

    records = get_attendance_by_date(date_str)

    if records:
        df = pd.DataFrame(records, columns=["Person Name", "Time Recorded"])
        df["Status"] = "Present"
        df.index = df.index + 1

        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name=f"attendance_{date_str}.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info(f"No attendance logs found for **{date_str}**.")


# =============================================================================
# Tab 4: Registered People Management
# =============================================================================
with tab4:
    st.header("Registered People Directory")
    st.caption("Manage registered profiles, inspect face crops, and remove records.")

    all_people = get_all_people()

    if all_people:
        for person_id, person_name in all_people:
            person_dir = os.path.join(PROJECT_ROOT, "data", "faces", person_name)
            img_count = len(os.listdir(person_dir)) if os.path.exists(person_dir) else 0

            with st.expander(f"👤 **{person_name}** ({img_count} images)"):
                c1, c2 = st.columns([3, 1])

                with c1:
                    if os.path.exists(person_dir):
                        sample_images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")][:5]
                        if sample_images:
                            img_cols = st.columns(len(sample_images))
                            for idx, img_file in enumerate(sample_images):
                                img_path = os.path.join(person_dir, img_file)
                                img_cols[idx].image(img_path, width=80)

                with c2:
                    if st.button(f"🗑️ Delete {person_name}", key=f"del_{person_id}"):
                        delete_person_by_name(person_name)
                        if os.path.exists(person_dir):
                            shutil.rmtree(person_dir)
                        st.cache_data.clear()
                        st.success(f"Deleted **{person_name}** successfully.")
                        st.rerun()
    else:
        st.info("No registered people found in the database. Use the 'Register New Person' tab to add people.")
