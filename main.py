import asyncio
import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# Import your modified modules (Refactored from your scripts)
# from agents.face_sim import FaceAgent
# from agents.fb_detect import DocAgent
# from agents.entity import EntityAgent
from scoring import VerificationScorer

# --- MOCK CLASSES (Replace these with your actual imports) ---
# I am writing wrappers here assuming you structure your existing files as classes
class FaceAgent:
    def compare(self, img1_url, img2_url):
        # Your logic from face_sim.py
        # Return: {"score": 85.5} or None
        pass 

class DocAgent:
    def detect_and_crop(self, front_url, back_url):
        # Your logic from fb_detect.py
        # Logic: 
        # 1. Download images
        # 2. Run YOLO (model 1)
        # 3. IF Front AND Back detected:
        #       Crop the Front Card area
        #       Return {"status": "success", "cropped_front_img": numpy_array}
        # 4. ELSE:
        #       Return {"status": "failed"}
        pass

class EntityAgent:
    def extract(self, image_array):
        # Your logic from entity.py
        # Logic:
        # 1. Run YOLO (model 2 - entities) on the cropped image
        # 2. Run OCR
        # 3. Return {"aadharnumber": "...", "dob": "...", "gender": "...", "age_status": "..."}
        pass
# -----------------------------------------------------------

app = FastAPI()
scorer = VerificationScorer()

# Initialize Agents (Load models globally on startup)
face_agent = FaceAgent()
doc_agent = DocAgent()
entity_agent = EntityAgent()
thread_pool = ThreadPoolExecutor(max_workers=4)

class VerificationRequest(BaseModel):
    selfie_url: str
    front_aadhaar_url: str
    back_aadhaar_url: str
    expected_gender: str = None # Optional

async def run_in_pool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args)

@app.post("/verify")
async def verify_user(data: VerificationRequest):
    """
    Main Verification Orchestrator
    """
    
    # --- PARALLEL EXECUTION START ---
    
    # Task 1: Face Similarity (Selfie vs Front Url)
    # We pass the raw Front URL because Face model handles full images well
    task_face = run_in_pool(face_agent.compare, data.selfie_url, data.front_aadhaar_url)
    
    # Task 2: Document Check -> Entity Extraction (The Pipeline)
    async def doc_entity_pipeline():
        # Step A: Detect Front/Back
        doc_result = await run_in_pool(doc_agent.detect_and_crop, data.front_aadhaar_url, data.back_aadhaar_url)
        
        if doc_result["status"] == "failed":
            return {"status": "failed", "reason": "Front or Back Aadhaar not detected"}
            
        # Step B: Entity Extraction (Only if Doc check passed)
        cropped_front = doc_result["cropped_front_img"]
        entity_result = await run_in_pool(entity_agent.extract, cropped_front)
        
        return {"status": "success", "data": entity_result}

    # Run both branches concurrently
    face_result, pipeline_result = await asyncio.gather(task_face, doc_entity_pipeline())
    
    # --- PARALLEL EXECUTION END ---

    # --- AGGREGATION & SCORING ---
    
    # 1. Check for Critical Failures
    if pipeline_result["status"] == "failed":
        return {
            "verified": False,
            "final_score": 0,
            "status": "REJECTED",
            "reason": pipeline_result["reason"]
        }

    # 2. Calculate Score
    entity_data = pipeline_result["data"]
    
    # Prepare data for scorer
    scoring_result = scorer.calculate_score(
        face_data=face_result if face_result else {"similarity": 0},
        entity_data=entity_data,
        expected_gender=data.expected_gender
    )
    
    return {
        "verified": scoring_result["status"] == "APPROVED",
        "final_score": scoring_result["total_score"],
        "status": scoring_result["status"],
        "details": {
            "scores": scoring_result["breakdown"],
            "extracted_data": {
                "aadhaar": entity_data.get("aadharnumber"),
                "dob": entity_data.get("dob"),
                "gender": entity_data.get("gender")
            },
            "rejection_reasons": scoring_result["rejection_reasons"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)