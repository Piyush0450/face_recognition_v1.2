"""
FaceRecognition - Stage 3: Multi-Face Detection Entry Point

This script detects multiple faces in real-time from the webcam feed using OpenCV's
Haar Cascade classifier, draws green bounding boxes with an "Unknown" label above
each detected face, and displays the total face count on screen.
"""

import os
import sys
import cv2


def load_face_cascade() -> cv2.CascadeClassifier:
    """Load the pre-trained Haar Cascade classifier for frontal face detection."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from: {cascade_path}")
        sys.exit(1)

    return face_cascade


def main() -> None:
    """Initialize webcam feed, perform multi-face detection, render labels, and handle cleanup."""
    # 1. Load the Haar Cascade face detector
    print("Loading Haar Cascade face detector...")
    face_cascade = load_face_cascade()

    # 2. Open default webcam (device 0)
    print("Opening default webcam (device 0)...")
    cap = cv2.VideoCapture(0)

    # 3. Handle error if the camera cannot be opened
    if not cap.isOpened():
        print("Error: Unable to access webcam. Please check camera connection or permissions.")
        sys.exit(1)

    # 4. Print webcam resolution (width x height) to terminal once
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened successfully. Resolution: {width}x{height}")
    print("Press 'q' in the video window to quit.")

    window_name = "FaceRecognition - Stage 3"
    last_face_count = -1  # Track face count to log changes to the terminal

    # 5. Process video frames inside a try/finally block to guarantee cleanup
    try:
        while True:
            # Read a single frame from the camera
            ret, frame = cap.read()

            # Handle frame capture errors
            if not ret or frame is None:
                print("Error: Failed to capture video frame from camera.")
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect all faces present in the grayscale frame
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            num_faces = len(faces)

            # Log to terminal only when detected face count changes
            if num_faces != last_face_count:
                print(f"Faces detected: {num_faces}")
                last_face_count = num_faces

            # Loop over all detected faces
            for (x, y, w, h) in faces:
                # 6. Draw green rectangle around each detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 7. Write "Unknown" label above each bounding box in green
                label_y = y - 10 if y - 10 > 20 else y + 20
                cv2.putText(
                    frame,
                    "Unknown",
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 8. Display stage title and total face count overlay in top-left corner
            cv2.putText(
                frame,
                "Stage 3: Multi-Face Detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Faces: {num_faces}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Display the frame in a window titled "FaceRecognition - Stage 3"
            cv2.imshow(window_name, frame)

            # Exit loop if user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User pressed 'q'. Exiting...")
                break

    finally:
        # Always release camera hardware and destroy all OpenCV windows cleanly
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed cleanly.")


if __name__ == "__main__":
    main()
