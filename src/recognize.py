"""
FaceRecognition - Stage 8: Threshold Calibration & Recognition Entry Point

This script performs real-time face recognition, allows live tuning of the cosine
similarity threshold using keyboard shortcuts ('t' to increase, 'y' to decrease, 's' to save),
provides a Calibration Mode ('c') for analyzing similarity scores of strangers/known faces,
and persists threshold configuration to config.py.
"""

import os
import pickle
import sys
import time
import warnings
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Suppress deprecation warnings from scikit-image inside InsightFace
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to sys.path to import config.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import load_threshold, save_threshold


def get_execution_context() -> tuple[int, str]:
    """Detect whether GPU (CUDA) or CPU is available for ONNX Runtime."""
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("GPU acceleration detected (CUDAExecutionProvider). Using GPU (ctx_id=0).")
        return 0, "GPU"
    else:
        print("GPU acceleration not found. Falling back to CPU (ctx_id=-1).")
        return -1, "CPU"


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Apply L2 normalization to a 1D numpy vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_embeddings(embeddings_path: str) -> dict[str, np.ndarray]:
    """Load known face embeddings dictionary from a pickle file."""
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file '{embeddings_path}' not found.")
        print("Please run Stage 6 (python src/encode_faces.py) to generate embeddings first.")
        sys.exit(1)

    with open(embeddings_path, "rb") as f:
        known_embeddings = pickle.load(f)

    if not known_embeddings:
        print(f"Error: Embeddings file '{embeddings_path}' is empty.")
        sys.exit(1)

    return known_embeddings


def main() -> None:
    """Initialize webcam, perform face recognition with live threshold tuning and calibration."""
    embeddings_file = os.path.join(project_root, "data", "embeddings.pkl")
    models_dir = os.path.join(project_root, "models")

    # 1. Load known face embeddings from disk
    print("Loading known face embeddings...")
    known_embeddings = load_embeddings(embeddings_file)
    print(f"Loaded {len(known_embeddings)} known person(s): {', '.join(known_embeddings.keys())}")

    # 2. Load recognition threshold from config.py
    current_threshold = load_threshold()
    print(f"Loaded similarity threshold from config: {current_threshold:.2f}")

    # 3. Detect hardware context (GPU vs CPU)
    ctx_id, device_type = get_execution_context()

    # 4. Initialize InsightFace FaceAnalysis model
    print("Loading InsightFace (buffalo_l) recognition model...")
    app = FaceAnalysis(name="buffalo_l", root=models_dir)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("InsightFace model ready.")

    # 5. Open default webcam (device 0)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access webcam. Please check camera connection or permissions.")
        sys.exit(1)

    print("\n--- CONTROLS ---")
    print(" 't' : Increase threshold (+0.05)")
    print(" 'y' : Decrease threshold (-0.05)")
    print(" 's' : Save threshold to config.py")
    print(" 'c' : Toggle Calibration Mode")
    print(" 'q' : Quit\n")

    window_name = "FaceRecognition - Stage 8"
    calibration_mode = False

    # Variables for FPS calculation and log throttling
    prev_time = time.time()
    fps = 0.0
    last_logged_scores = ""

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Error: Failed to capture frame from webcam.")
                break

            # Detect faces using InsightFace
            detected_faces = app.get(frame)

            total_faces = len(detected_faces)
            known_count = 0
            unknown_count = 0
            log_entries = []

            # 6. Process each detected face
            for face in detected_faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                # Extract and L2-normalize live face embedding
                raw_embedding = face.embedding
                norm_embedding = l2_normalize(raw_embedding)

                # Compare against known embeddings using Cosine Similarity (dot product)
                best_match_name = "Unknown"
                best_score = 0.0

                for name, known_vec in known_embeddings.items():
                    score = float(np.dot(norm_embedding, known_vec))
                    if score > best_score:
                        best_score = score
                        best_match_name = name

                # Evaluate against current threshold
                is_match = best_score >= current_threshold

                # Formulate log entry string for terminal output
                match_status = "MATCH" if is_match else "NO MATCH"
                log_entries.append(
                    f"Best match: {best_match_name} ({best_score:.2f}) | Threshold: {current_threshold:.2f} → {match_status}"
                )

                if is_match:
                    label = f"{best_match_name} {int(best_score * 100)}%"
                    color = (0, 255, 0)  # Green for recognized face
                    known_count += 1
                else:
                    label = f"Unknown ({int(best_score * 100)}%)" if calibration_mode else "Unknown"
                    color = (0, 0, 255)  # Red for unknown face
                    unknown_count += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Render text label above box
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

            # Print similarity score diagnostic to console when detection results change
            current_log = " | ".join(log_entries)
            if log_entries and current_log != last_logged_scores:
                for entry in log_entries:
                    prefix = "[CALIBRATION] " if calibration_mode else ""
                    print(f"{prefix}{entry}")
                last_logged_scores = current_log

            # 7. Calculate real-time FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = current_time

            # 8. Display on-screen overlays
            # Stage Header
            cv2.putText(
                frame,
                "Stage 8: Recognition & Calibration",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # FPS and Threshold Info
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Threshold: {current_threshold:.2f} (t=+, y=-, s=Save, c=Calibrate)",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Detection Counts
            cv2.putText(
                frame,
                f"Faces: {total_faces} | Known: {known_count} | Unknown: {unknown_count}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Calibration Mode Banner
            if calibration_mode:
                cv2.putText(
                    frame,
                    "[CALIBRATION MODE ACTIVE]",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)

            # 9. Handle Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting recognition...")
                break
            elif key == ord("t"):
                # Increase threshold by 0.05 (max 0.80)
                current_threshold = min(0.80, round(current_threshold + 0.05, 2))
                print(f"Threshold increased to: {current_threshold:.2f}")
            elif key == ord("y"):
                # Decrease threshold by 0.05 (min 0.30)
                current_threshold = max(0.30, round(current_threshold - 0.05, 2))
                print(f"Threshold decreased to: {current_threshold:.2f}")
            elif key == ord("s"):
                # Save current threshold to config.py
                if save_threshold(current_threshold):
                    print(f"SUCCESS: Saved threshold {current_threshold:.2f} to config.py")
            elif key == ord("c"):
                # Toggle calibration mode
                calibration_mode = not calibration_mode
                status = "ENABLED" if calibration_mode else "DISABLED"
                print(f"Calibration Mode {status}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed cleanly.")


if __name__ == "__main__":
    main()
