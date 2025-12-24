import os
import cv2
import json
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from entity import EntityAgent
import pickle
import 
# ------------------ CONFIG ------------------

UPLOAD_ROOT = "uploads"
MODEL_PATH = "../models/best.pt"
CONFIDENCE_THRESHOLD = 0.15

# ------------------ APP ------------------

app = FastAPI(title="Aadhaar Entity Extraction API")

# Initialize agent once
agent = EntityAgent(model_path=MODEL_PATH)

# ------------------ REQUEST MODEL ------------------

class AadhaarRequest(BaseModel):
    user_id: str
    front_image_url: HttpUrl
    back_image_url: HttpUrl

# ------------------ UTILITIES ------------------

def download_image(url: str, save_path: str):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    content_type = r.headers.get("content-type", "")
    if "image" not in content_type:
        raise ValueError("URL does not point to an image")
    with open(save_path, "wb") as f:
        f.write(r.content)

def draw_boxes(image, detections):
    for d in detections:
        # detection items expected to be dicts with bbox and class_name or label
        bbox = d.get("bbox") or d.get("bbox_xy") or d.get("xyxy")
        label = d.get("class_name") or d.get("label") or d.get("name", "entity")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def normalize_detection_list(all_detections):
    """
    all_detections is expected to be the structure returned by
    agent.detect_entities_in_image(image_array, confidence_threshold).
    This helper picks the first card entry and returns its detection list.
    """
    if not all_detections:
        return [], None  # no detections, no image

    # Pick the first card key (e.g. "memory_crop" or filename stem)
    first_key = next(iter(all_detections.keys()))
    card = all_detections[first_key]
    detections = card.get("detections", []) or []

    # Normalize each detection to have 'class_name' and 'bbox'
    norm = []
    for d in detections:
        bbox = d.get("bbox") or d.get("xyxy") or d.get("xyxy0") or d.get("xyxy")
        cls = d.get("class_name") or d.get("label") or d.get("name")
        conf = d.get("confidence") or d.get("conf")
        norm.append({
            "class_name": cls if cls else "unknown",
            "bbox": list(map(int, bbox)) if bbox else None,
            "confidence": float(conf) if conf is not None else None
        })
    return norm, card.get("card_image", None)

def process_side(side: str, image_url: str, user_dir: str):
    """
    Downloads image, runs detection helper to get boxes + crops,
    saves annotated image and crops, and calls agent.extract_from_file() to get final OCR data.
    Returns a dictionary with saved paths and the agent result.
    """
    side_dir = os.path.join(user_dir, side)
    crops_dir = os.path.join(side_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    original_path = os.path.join(side_dir, "original.jpg")
    annotated_path = os.path.join(side_dir, "annotated.jpg")

    # 1) Download image
    download_image(image_url, original_path)

    # 2) Prepare image array
    img = cv2.imread(original_path)
    if img is None:
        raise RuntimeError(f"Failed to load downloaded image at {original_path}")

    # 3) Use agent's detection helper (works with numpy array)
    #    This returns the same structure as agent.detect_entities_in_image expects.
    try:
        all_detections = agent.detect_entities_in_image(img, CONFIDENCE_THRESHOLD)
    except Exception:
        # Fallback: try passing a file path (some agent implementations accept path)
        all_detections = agent.detect_entities_in_image(original_path, CONFIDENCE_THRESHOLD)

    detections, card_image = normalize_detection_list(all_detections)

    saved_crops = []
    # 4) Crop each detection and save
    for idx, det in enumerate(detections):
        bbox = det.get("bbox")
        label = det.get("class_name") or "entity"
        if not bbox:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        # sanitize bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        crop_name = f"{label}_{idx}.jpg"
        crop_path = os.path.join(crops_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        det["crop_path"] = crop_path
        saved_crops.append(crop_path)

    # 5) Draw and save annotated image (use detections normalized)
    annotated_img = draw_boxes(img.copy(), [{"bbox": d["bbox"], "label": d.get("class_name", "") , "class_name": d.get("class_name")} for d in detections])
    cv2.imwrite(annotated_path, annotated_img)

    # 6) Call agent.extract_from_file(...) to perform OCR + final field extraction.
    #    Many Agent implementations accept a file path and will run end-to-end.
    agent_result = agent.extract_from_file(original_path)

    return {
        "original_path": original_path,
        "annotated_path": annotated_path,
        "crops": saved_crops,
        "agent_result": agent_result,
        "detections": detections
    }

def pick_final_fields_from_agent_result(agent_result):
    """
    Agent may return different shapes. This helper searches common keys.
    Prefer front_result['data'] or front_result['final'].
    """
    if not agent_result:
        return {}

    # If agent_result is the nested wrapper {"success": True, "data": {...}}
    if isinstance(agent_result, dict):
        if agent_result.get("success") and isinstance(agent_result.get("data"), dict):
            return agent_result["data"]
        if isinstance(agent_result.get("final"), dict):
            return agent_result["final"]
        # some implementations return 'data' directly
        if isinstance(agent_result.get("data"), dict):
            return agent_result["data"]
        # maybe agent_result itself is the data
        keys = set(agent_result.keys())
        if {"aadharnumber", "dob", "gender"} & keys:
            return agent_result

    return {}

# ------------------ API ------------------

@app.post("/extract/aadhaar")
async def extract_aadhaar(payload: AadhaarRequest):
    user_dir = os.path.join(UPLOAD_ROOT, payload.user_id)
    os.makedirs(user_dir, exist_ok=True)

    try:
        # Process front and back sides (saves annotated images and crops)
        front_info = process_side("front", payload.front_image_url, user_dir)
        back_info  = process_side("back", payload.back_image_url, user_dir)

        # Get final fields from agent outputs (prefer front, fallback to back)
        front_fields = pick_final_fields_from_agent_result(front_info.get("agent_result"))
        back_fields  = pick_final_fields_from_agent_result(back_info.get("agent_result"))

        final = {}
        # prefer front fields if present, otherwise back
        final["aadharnumber"] = front_fields.get("aadharnumber") or back_fields.get("aadharnumber") or front_fields.get("aadhaar_number") or back_fields.get("aadhaar_number")
        final["dob"] = front_fields.get("dob") or back_fields.get("dob") or front_fields.get("date_of_birth") or back_fields.get("date_of_birth")
        final["gender"] = front_fields.get("gender") or back_fields.get("gender")

        # Also pull additional metadata if available
        final["age"] = front_fields.get("age") or back_fields.get("age")
        final["age_status"] = front_fields.get("age_status") or back_fields.get("age_status")
        final["aadhar_status"] = front_fields.get("aadhar_status") or back_fields.get("aadhar_status")

        response = {
            "status": "success",
            "aadhaar_number": final.get("aadharnumber"),
            "dob": final.get("dob"),
            "gender": final.get("gender"),
            "age": final.get("age"),
            "age_status": final.get("age_status"),
            "aadhar_status": final.get("aadhar_status"),
            "files": {
                "front_original": front_info.get("original_path"),
                "front_annotated": front_info.get("annotated_path"),
                "back_original": back_info.get("original_path"),
                "back_annotated": back_info.get("annotated_path"),
                "front_crops": front_info.get("crops", []),
                "back_crops": back_info.get("crops", [])
            }
        }

        # Save final JSON for audit
        with open(os.path.join(user_dir, "result.json"), "w") as f:
            json.dump(response, f, indent=4)

        return response

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ------------------ RUN ------------------

if __name__ == "__main__":
    uvicorn.run("entityCheck:app", host="0.0.0.0", port=8101)
