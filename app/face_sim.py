import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAgent:
    def __init__(self, model_name='buffalo_l'):
        print(f"Loading Face Model ({model_name})...")
        # Initialize model once on startup
        self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None

        faces = self.app.get(img)
        
        if len(faces) == 0:
            print(f"No face detected in {img_path}")
            return None
        return faces[0].embedding

    def compare(self, img1_path, img2_path):
        """
        Compares two local image files.
        Returns: float score (0-100) or None if failed.
        """
        emb1 = self.get_embedding(img1_path)
        emb2 = self.get_embedding(img2_path)

        if emb1 is None or emb2 is None:
            return 0.0 # Return 0 similarity if detection fails
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        score = similarity * 100
        return float(max(0, min(100, score)))
    
if __name__ == "__main__":
    agent = FaceAgent()
    # print(agent.compare("test_selfie.jpg", "test_card.jpg"))