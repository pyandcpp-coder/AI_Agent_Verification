import cv2
import numpy as np
import os
import tempfile
import requests
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

def download_image(url_or_path):
    """Download image from URL to temp file, or return path if already local."""
    if url_or_path.startswith(('http://', 'https://')):
        try:
            response = requests.get(url_or_path, timeout=10)
            response.raise_for_status()
            
            # Create temp file with appropriate extension
            ext = os.path.splitext(url_or_path)[1] or '.jpg'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name, True  # True indicates it's a temp file to delete
        except Exception as e:
            print(f"Error downloading image from {url_or_path}: {e}")
            return None, False
    else:
        # It's already a local path
        return url_or_path, False

def compare_faces(img1_path, img2_path):
    print("Processing...")
    
    # Download images if they're URLs
    local_img1, delete_img1 = download_image(img1_path)
    local_img2, delete_img2 = download_image(img2_path)
    
    if local_img1 is None or local_img2 is None:
        return None
    
    try:
        emb1 = get_embedding(local_img1)
        emb2 = get_embedding(local_img2)

        if emb1 is None or emb2 is None:
            return None
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
        
        return float(score)
    finally:
        # Clean up temp files
        if delete_img1 and os.path.exists(local_img1):
            os.unlink(local_img1)
        if delete_img2 and os.path.exists(local_img2):
            os.unlink(local_img2)

compare_faces('https://cdn.qoneqt.com/uploads/113711/aadhar_front_113711.jpg', 'https://cdn.qoneqt.com/uploads/113711/aadhar_front_113711.jpg')