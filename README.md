# FaceRecognition - Stage 1

Real-time multi-person face recognition and attendance system (Stage 1 Skeleton & Webcam Verification).

## Project Overview
Stage 1 sets up the minimal project folder structure, pins core dependencies in `requirements.txt`, and verifies OpenCV webcam capture functionality.

---

## Directory Structure
```text
FaceRecognition/
│
├── data/
│   └── faces/              → Storage for registered face images (future stages)
│
├── models/                 → Storage for InsightFace models (future stages)
│
├── database/               → Storage for SQLite attendance database (future stages)
│
├── src/
│   └── __init__.py         → Package root for source modules
│
├── requirements.txt        → Pinned project dependencies
├── main.py                 → Stage 1 entry point (Webcam test script)
└── README.md               → Documentation
```

---

## Prerequisites
- Python 3.10 or 3.11
- System camera / webcam

---

## How to Run

1. **Open your terminal** in the project directory:
   ```bash
   cd "d:\projects\face recognition"
   ```

2. **Create a virtual environment**:
   - **Windows**:
     ```cmd
     python -m venv venv
     ```
   - **macOS / Linux**:
     ```bash
     python3 -m venv venv
     ```

3. **Activate the virtual environment**:
   - **Windows (Command Prompt)**:
     ```cmd
     venv\Scripts\activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run main.py**:
   ```bash
   python main.py
   ```

6. **Quit the camera feed**:
   - Press the key **`q`** while focusing on the video window.
"face_recognition_1.2" 
