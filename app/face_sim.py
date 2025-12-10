import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceAgent:
    def __init__(self, model_name='buffalo_l'):
        print(f"Loading Face Model ({model_name})...")
        # Load model once at startup with genderage module
        self.app = FaceAnalysis(
            name=model_name, 
            allowed_modules=['detection', 'recognition', 'genderage'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, img_path):
        # Read local file directly
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None

        faces = self.app.get(img)
        
        if len(faces) == 0:
            print(f"No face detected in {img_path}")
            return None
        
        face = faces[0]
        gender = "Male" if face.gender == 1 else "Female"
        
        return {
            'embedding': face.embedding,
            'gender': gender
        }

    def compare(self, img1_path, img2_path):
        # Accepts LOCAL PATHS now
        result1 = self.get_embedding(img1_path)
        result2 = self.get_embedding(img2_path)

        if result1 is None or result2 is None:
            return {'score': 0.0, 'gender_img1': None, 'gender_img2': None}

        emb1 = result1['embedding']
        emb2 = result2['embedding']

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        score = similarity * 100
        
        return {
            'score': float(max(0, min(100, score))),
            'gender_img1': result1['gender'],
            'gender_img2': result2['gender']
        }
    