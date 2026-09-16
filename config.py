"""
FaceRecognition Configuration Module

Stores configurable parameters such as recognition thresholds.
"""

import os

# Cosine similarity threshold for face recognition (valid range: 0.30 to 0.80)
SIMILARITY_THRESHOLD = 0.50


def load_threshold() -> float:
    """Load the recognition threshold from config, defaulting to 0.50 if not set."""
    try:
        import config
        threshold = getattr(config, "SIMILARITY_THRESHOLD", 0.50)
        return max(0.30, min(0.80, float(threshold)))
    except Exception:
        return 0.50


def save_threshold(new_threshold: float) -> bool:
    """Save the updated threshold to this config.py file."""
    clamped_threshold = max(0.30, min(0.80, round(new_threshold, 2)))
    file_path = os.path.abspath(__file__)
    try:
        content = f'''"""
FaceRecognition Configuration Module

Stores configurable parameters such as recognition thresholds.
"""

import os

# Cosine similarity threshold for face recognition (valid range: 0.30 to 0.80)
SIMILARITY_THRESHOLD = {clamped_threshold:.2f}
'''
        with open(file_path, "w") as f:
            f.write(content)
        return True
    except Exception as err:
        print(f"Error saving threshold to config.py: {err}")
        return False
