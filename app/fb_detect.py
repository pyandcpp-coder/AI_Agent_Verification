from fastapi import FastAPI, HTTPException
from ultralytics import YOLO
import requests
import numpy as np
import cv2
import uvicorn

app = FastAPI()

# Load model
model = YOLO("models/best4.pt")

# Configuration
CONFIDENCE_THRESHOLD = 0.70

def next_process(image_array):
    """
    Placeholder for the next step (e.g., OCR, cropping, saving).
    Receives the numpy/cv2 image array of the Front Aadhaar.
    """
    # Example logic: Get image dimensions to prove we have the image
    height, width, _ = image_array.shape
    print(f"Processing Front Aadhaar. Size: {width}x{height}")
    
    # Return whatever result this process generates
    return {
        "status": "success", 
        "message": "Front Aadhaar processed successfully",
        "image_shape": [height, width]
    }

def process_image(url):
    """
    Downloads image, runs detection, and filters by threshold.
    Returns: (image_array, detected_label, confidence_score)
    """
    try:
        # 1. Download image
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_bytes = resp.content

        # 2. Convert bytes to CV2 image
        img = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        if img is None:
            return None, None, 0.0

        # 3. Detect
        results = model(img)

        best_label = None
        max_conf = 0.0

        # 4. Check results and filter by Threshold
        for box in results[0].boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = model.names[cls_id]

            # Logic: We want the highest confidence detection that passes the threshold
            if conf >= CONFIDENCE_THRESHOLD and conf > max_conf:
                max_conf = conf
                best_label = label

        return img, best_label, max_conf

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, 0.0


@app.get("/aadhaar/detect")
async def detect_and_process(front_url: str, back_url: str):
    
    # Process both images to find out what they are
    img1, label1, conf1 = process_image(front_url)
    img2, label2, conf2 = process_image(back_url)

    # Check if images failed to load
    if img1 is None or img2 is None:
        raise HTTPException(status_code=400, detail="Failed to download or decode one of the images.")

    detected_front_img = None
    detection_info = {}

    # --- LOGIC TO IDENTIFY FRONT AADHAAR ---
    
    # Note: Adjust "front" and "back" strings to match exactly what your model.names returns
    # We convert to lower() just to be safe.
    
    # Check Image 1
    if label1 and "front" in label1.lower():
        detected_front_img = img1
        detection_info['url_1'] = f"Detected Front ({conf1})"
    elif label1:
        detection_info['url_1'] = f"Detected {label1} ({conf1})"
    else:
        detection_info['url_1'] = "No detection above threshold"

    # Check Image 2 (Only if Image 1 wasn't already the front, or if we want to pick the best one)
    if label2 and "front" in label2.lower():
        # If we already found a front in img1, usually we pick the higher confidence
        if detected_front_img is not None:
            if conf2 > conf1:
                detected_front_img = img2
                detection_info['url_2'] = f"Detected Front ({conf2}) - SELECTED"
                detection_info['url_1'] = f"Detected Front ({conf1}) - IGNORED (Lower Conf)"
        else:
            detected_front_img = img2
            detection_info['url_2'] = f"Detected Front ({conf2})"
    elif label2:
        detection_info['url_2'] = f"Detected {label2} ({conf2})"
    else:
        detection_info['url_2'] = "No detection above threshold"


    # --- FINAL DECISION ---

    if detected_front_img is not None:
        # Pass ONLY the front image to the next process
        next_step_result = next_process(detected_front_img)
        
        return {
            "detection_summary": detection_info,
            "next_process_result": next_step_result
        }
    else:
        # If no front card was found above 0.70 threshold
        return {
            "status": "failed",
            "message": "No valid Aadhaar Front detected above 0.70 threshold.",
            "detection_summary": detection_info
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)