"""
FaceRecognition - Stage 6: Generate Face Embeddings

This script uses InsightFace (buffalo_l) to extract 512-dimensional facial embeddings
from registered face images in data/faces/<Name>/, L2-normalizes them, averages them per person,
and saves the result to data/embeddings.pkl using pickle.
"""

import argparse
import os
import pickle
import sys
import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


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
    """Apply L2 normalization to a 1D or 2D numpy array embedding vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def main() -> None:
    """Generate 512-dim embeddings for all registered persons in data/faces/."""
    # 1. Parse command-line arguments (--rebuild flag)
    parser = argparse.ArgumentParser(description="Extract and save 512-dim face embeddings using InsightFace.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing data/embeddings.pkl before generating new embeddings.",
    )
    args = parser.parse_args()

    # Determine project root and relevant directory paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faces_dir = os.path.join(project_root, "data", "faces")
    models_dir = os.path.join(project_root, "models")
    embeddings_file = os.path.join(project_root, "data", "embeddings.pkl")

    # Handle --rebuild flag: delete old embeddings.pkl if present
    if args.rebuild and os.path.exists(embeddings_file):
        print(f"Deleting existing embeddings file: {embeddings_file}")
        os.remove(embeddings_file)

    # 2. Check if data/faces/ directory exists and contains registered people
    if not os.path.exists(faces_dir):
        print(f"Error: Directory '{faces_dir}' does not exist. Please run registration (Stage 5) first.")
        sys.exit(1)

    person_folders = [
        d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))
    ]

    if not person_folders:
        print(f"Error: No registered person folders found in '{faces_dir}'. Please register at least one face.")
        sys.exit(1)

    print(f"Found {len(person_folders)} registered person(s): {', '.join(person_folders)}")

    # 3. Detect hardware context (GPU vs CPU)
    ctx_id, device_type = get_execution_context()

    # 4. Load InsightFace FaceAnalysis model (buffalo_l)
    print("Loading InsightFace (buffalo_l) model...")
    # root parameter ensures model weights are saved/loaded under the models/ directory
    app = FaceAnalysis(name="buffalo_l", root=models_dir)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print("InsightFace model initialized successfully.")

    known_embeddings = {}

    # 5. Process each registered person directory
    for person_name in person_folders:
        person_dir = os.path.join(faces_dir, person_name)
        image_files = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        total_images = len(image_files)
        if total_images == 0:
            print(f"{person_name}: 0 images found. Skipping.")
            continue

        person_embeddings = []

        # Process each face image
        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Detect faces using InsightFace
            detected_faces = app.get(img)

            # Skip if zero or more than one face is detected in the crop
            if len(detected_faces) != 1:
                continue

            # Extract the 512-dimensional embedding vector
            raw_embedding = detected_faces[0].embedding

            # L2-normalize the individual embedding
            normalized_embedding = l2_normalize(raw_embedding)
            person_embeddings.append(normalized_embedding)

        # 6. Average and L2-normalize all valid embeddings for this person
        if person_embeddings:
            # Calculate element-wise mean vector across all valid images
            mean_embedding = np.mean(person_embeddings, axis=0)

            # Re-normalize the averaged 512-dim embedding vector
            final_embedding = l2_normalize(mean_embedding)

            known_embeddings[person_name] = final_embedding
            print(f"{person_name}: {len(person_embeddings)}/{total_images} images used")
        else:
            print(f"{person_name}: 0/{total_images} images used (no valid faces detected)")

    # 7. Save the embeddings dictionary to data/embeddings.pkl using pickle
    if known_embeddings:
        os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
        with open(embeddings_file, "wb") as f:
            pickle.dump(known_embeddings, f)

        print(f"Saved {len(known_embeddings)} people to data/embeddings.pkl")
    else:
        print("Warning: No valid face embeddings were generated. embeddings.pkl not created.")


if __name__ == "__main__":
    main()
