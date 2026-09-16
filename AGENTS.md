# AGENTS.md

## Project
FaceRecognition – real-time multi-person face recognition and attendance

## Stack
Python 3.10/3.11, OpenCV, InsightFace, SQLite, numpy

## Commands
- Create venv: python -m venv venv
- Activate Windows: venv\Scripts\activate
- Activate Mac/Linux: source venv/bin/activate
- Install: pip install -r requirements.txt
- Run: python main.py

## Rules
- Build one stage at a time. Never skip ahead.
- Do not add dependencies without asking.
- Keep code beginner-friendly with comments.
- Use try/finally to release the camera.
- Cross-platform (Windows, Linux, macOS).
- Do not change folder structure without asking.

## Structure
- data/faces/<Name>/ → registered face images
- models/ → InsightFace models
- database/ → attendance.db
- src/ → source code
- main.py → entry point

## Stages
1. Environment + webcam
2. Detect one face
3. Detect multiple faces
4. Draw green/red boxes
5. Register people
6. Generate embeddings
7. Recognition
8. Unknown-person threshold
9. SQLite database
10. Attendance
11. UI/dashboard
12. Optimize + interview prep