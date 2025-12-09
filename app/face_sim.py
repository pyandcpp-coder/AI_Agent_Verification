import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) 

def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return None

    faces = app.get(img)
    
    if len(faces) == 0:
        print(f"No face detected in {img_path}")
        return None
    return faces[0].embedding

def compare_faces(img1_path, img2_path):
    print("Processing...")
    
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    if emb1 is None or emb2 is None:
        return
    # cosine sim
    # Formula: (A . B) / (||A|| * ||B||)
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    score = similarity * 100
    # 0.4 is a common safe threshold for "Same Person"
    threshold = 30 # relaxed threshold for demo, strict is ~40-50
    
    print(f"-----------------------------")
    print(f"Similarity Score: {score:.2f}%")
    
    if score > threshold:
        print("✅ RESULT: Same Person")
    else:
        print("❌ RESULT: Different People")
    print(f"-----------------------------")

compare_faces('images/t1a.jpg', 'images/t1f.jpg')