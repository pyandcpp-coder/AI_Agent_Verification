# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet")
PRIMARY_DETECTOR = os.getenv("PRIMARY_DETECTOR", "retinaface")
SECONDARY_DETECTOR = os.getenv("SECONDARY_DETECTOR", "mtcnn")

# --- Thresholds ---
USER_SIMILARITY_THRESHOLD = float(os.getenv("USER_SIMILARITY_THRESHOLD", 0.75))
EMPLOYEE_SIMILARITY_THRESHOLD = float(os.getenv("EMPLOYEE_SIMILARITY_THRESHOLD", 0.95))
FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", 0.95))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 40.0))
TERTIARY_DETECTOR = os.getenv("TERTIARY_DETECTOR", "dlib")
# --- File Paths ---
USER_EMBEDDINGS_FILE = os.getenv("USER_EMBEDDINGS_FILE", "user_embeddings.pkl")
EMPLOYEE_EMBEDDINGS_FILE = os.getenv("EMPLOYEE_EMBEDDINGS_FILE", "employee_embeddings.pkl")