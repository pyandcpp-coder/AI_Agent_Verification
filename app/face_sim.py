import cv2
import numpy as np
from deepface import DeepFace

class FaceAgent:
    def __init__(
        self,
        model_name: str = "Facenet512",
        detector_backend: str = "mtcnn"
    ):
        """
        model_name options:
        - ArcFace (recommended)
        - Facenet512
        - VGG-Face
        - OpenFace

        detector_backend options:
        - retinaface (best)
        - mtcnn
        - opencv
        - mediapipe
        """
        print(f"Loading DeepFace model ({model_name})...")
        self.model_name = model_name
        self.detector_backend = detector_backend

    def get_embedding(self, img_path: str):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None

        try:
            embeddings = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
        except Exception as e:
            print(f"No face detected in {img_path}: {e}")
            return None

        if not embeddings:
            return None

        return np.array(embeddings[0]["embedding"])

    def compare(self, img1_path: str, img2_path: str):
        """
        Compare two face images and return similarity score only.

        Returns:
            dict: {
                'score': float (0-100),
                'source': 'face_agent'
            }
        """
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)

        if emb1 is None or emb2 is None:
            return {
                "score": 0.0,
                "source": "face_agent"
            }

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        score = similarity * 100

        return {
            "score": float(max(0, min(100, score))),
            "source": "face_agent"
        }
