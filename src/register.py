"""
FaceRecognition - Stage 4: Face Registration

This script prompts the user for a person's name, creates a folder for them under
data/faces/<Name>/, opens the webcam, detects faces, crops face regions with a small
margin, and captures 20 face images to disk.
"""

import os
import re
import subprocess
import sys
import time
import cv2


def sanitize_name(name: str) -> str:
    """Sanitize user input string into a safe folder name."""
    # Replace spaces with underscores and remove non-alphanumeric characters (except underscores & hyphens)
    clean_name = name.strip().replace(" ", "_")
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "", clean_name)
    return clean_name


def load_face_cascade() -> cv2.CascadeClassifier:
    """Load the pre-trained Haar Cascade classifier for frontal face detection."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from: {cascade_path}")
        sys.exit(1)

    return face_cascade


def main() -> None:
    """Run interactive registration process to capture 20 face images."""
    # 1. Ask the user for the person's name
    raw_name = input("Enter person's name: ")
    person_name = sanitize_name(raw_name)

    if not person_name:
        print("Error: Invalid or empty name provided. Registration cancelled.")
        sys.exit(1)

    # Determine paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    person_dir = os.path.join(project_root, "data", "faces", person_name)

    # 2. Create data/faces/<PersonName>/ directory if it does not exist
    os.makedirs(person_dir, exist_ok=True)
    print(f"Saving face images to: {person_dir}")

    # 3. Load Haar Cascade face detector
    face_cascade = load_face_cascade()

    # 4. Open default webcam (device 0)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access webcam. Please check camera connection or permissions.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam ready ({frame_width}x{frame_height}). Look at the camera.")
    print("Press 'q' at any time to cancel registration.")

    target_count = 20
    captured_count = 0
    last_capture_time = 0.0
    delay_between_captures = 0.2  # 200 ms delay between captures

    window_name = "FaceRecognition - Register"

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Error: Failed to capture video frame.")
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in frame
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Check if exactly one face is detected
            if len(faces) == 1:
                (x, y, w, h) = faces[0]

                # Draw green bounding box around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Capture image if 0.2 seconds have passed since last capture
                current_time = time.time()
                if (current_time - last_capture_time >= delay_between_captures) and (captured_count < target_count):
                    captured_count += 1
                    last_capture_time = current_time

                    # 5. Crop the face region with 50% margin around bounding box to provide full facial context for InsightFace
                    margin_x = int(w * 0.50)
                    margin_y = int(h * 0.50)

                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(frame_width, x + w + margin_x)
                    y2 = min(frame_height, y + h + margin_y)

                    face_crop = frame[y1:y2, x1:x2]

                    # Save face crop image (1.jpg, 2.jpg, ... 20.jpg)
                    image_filename = f"{captured_count}.jpg"
                    image_path = os.path.join(person_dir, image_filename)
                    cv2.imwrite(image_path, face_crop)
                    print(f"Captured {captured_count}/{target_count}: {image_filename}")

            elif len(faces) > 1:
                # Warning if multiple faces are visible
                cv2.putText(
                    frame,
                    "Warning: Multiple faces detected! Show only 1 face.",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 6. Show progress text overlay on screen
            cv2.putText(
                frame,
                f"Registering: {person_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            status_text = f"Capturing {captured_count}/{target_count}" if captured_count < target_count else "Complete!"
            cv2.putText(
                frame,
                status_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Display frame in GUI window
            cv2.imshow(window_name, frame)

            # 7. Check completion or 'q' exit
            if captured_count >= target_count:
                print(f"Done. Registered {person_name}.")
                cv2.waitKey(500)
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Registration cancelled by user.")
                break

    finally:
        # Always release camera hardware and destroy windows cleanly
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed cleanly.")

    # Automatically trigger embedding generation & DB migration upon successful capture
    if captured_count >= target_count:
        print("\n--- AUTOMATIC EMBEDDING & DATABASE UPDATE ---")
        python_exe = sys.executable
        encode_script = os.path.join(project_root, "src", "encode_faces.py")
        migrate_script = os.path.join(project_root, "src", "migrate_to_db.py")

        res1 = subprocess.run([python_exe, encode_script])
        if res1.returncode == 0:
            res2 = subprocess.run([python_exe, migrate_script])
            if res2.returncode == 0:
                print(f"\n[SUCCESS] Registration, embedding generation, and database sync complete for '{person_name}'!")
            else:
                print("\n[ERROR] Failed during database migration.")
        else:
            print("\n[ERROR] Failed during face embedding extraction.")


if __name__ == "__main__":
    main()
