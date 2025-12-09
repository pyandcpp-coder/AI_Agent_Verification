import os
import shutil
import asyncio
import aiohttp
import aiofiles
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union

# --- IMPORT ACTUAL AGENTS ---
try:
    from app.face_sim import FaceAgent
    from app.fb_detect import DocAgent
    from app.entity import EntityAgent
    from scoring import VerificationScorer
except ImportError:
    # Fallback for flat directory structures
    from face_sim import FaceAgent
    from fb_detect import DocAgent
    from entity import EntityAgent
    from scoring import VerificationScorer

app = FastAPI()

# --- Global State ---
face_agent = None
doc_agent = None
entity_agent = None
scorer = None

TEMP_DIR = Path("temp")

@app.on_event("startup")
async def startup_event():
    """Initialize all heavy models once at startup to save RAM/Time."""
    global face_agent, doc_agent, entity_agent, scorer
    
    # Create temp dir if not exists
    TEMP_DIR.mkdir(exist_ok=True)

    print("--- STARTING SYSTEM INITIALIZATION ---")
    face_agent = FaceAgent()      # Loads InsightFace
    doc_agent = DocAgent()        # Loads YOLO for Doc Detection
    entity_agent = EntityAgent()  # Loads YOLO for Entities + Tesseract
    scorer = VerificationScorer() # Loads Scoring Logic
    print("--- SYSTEM READY ---")

# --- INPUT MODEL MATCHING YOUR SPECIFIC JSON FORMAT ---
class VerifyRequest(BaseModel):
    user_id: Union[int, str]
    dob: Optional[str] = None          # e.g., "06-12-2007"
    passport_first: str                # Front Aadhaar (URL or Local Path)
    passport_old: str                  # Back Aadhaar (URL or Local Path)
    selfie_photo: str                  # Selfie (URL or Local Path)
    gender: Optional[str] = None       # "Male/Female/Other"

async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
    """
    Hybrid Fetcher:
    - If source is http/https -> Downloads it.
    - If source is a local path -> Copies it.
    """
    try:
        if str(source).startswith(('http://', 'https://')):
            async with session.get(str(source), timeout=15) as resp:
                if resp.status == 200:
                    async with aiofiles.open(destination, 'wb') as f:
                        await f.write(await resp.read())
                    return True
                else:
                    print(f"Failed to download URL: {source} (Status: {resp.status})")
        else:
            # Assume local path (e.g., "uploads/38070/...")
            # We resolve it relative to the current working directory
            source_path = Path(source)
            if source_path.exists():
                shutil.copy(source_path, destination)
                return True
            else:
                print(f"Local file not found: {source}")
    except Exception as e:
        print(f"Error fetching {source}: {e}")
    return False

async def setup_files(user_id: str, req: VerifyRequest):
    """
    Sets up the temp environment: temp/{user_id}/...
    renaming files to standard names for processing.
    """
    task_dir = TEMP_DIR / str(user_id)
    
    # Clean existing directory if it exists to ensure fresh start
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Define standard filenames for internal processing
    # Structure: {user_id}/{user_id}_{type}.jpg
    paths = {
        "selfie": task_dir / f"{user_id}_selfie.jpg",
        "front": task_dir / f"{user_id}_aadhar_front.jpg",
        "back": task_dir / f"{user_id}_aadhar_back.jpg"
    }

    async with aiohttp.ClientSession() as session:
        # Fetch all 3 concurrently
        results = await asyncio.gather(
            fetch_file(session, req.selfie_photo, paths["selfie"]),
            fetch_file(session, req.passport_first, paths["front"]),
            fetch_file(session, req.passport_old, paths["back"])
        )

    # Check if all files were successfully retrieved
    if all(results):
        # Return string paths for OpenCV compatibility
        return {k: str(v) for k, v in paths.items()}, task_dir
    else:
        # Cleanup immediately if any file failed
        shutil.rmtree(task_dir)
        return None, None

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    """Full verification endpoint with detailed breakdown (for testing/debugging)"""
    user_id = str(req.user_id)
    print(f"[{user_id}] New Verification Request")

    # 1. Setup Files (Download/Copy)
    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "status": "FAILED", 
            "reason": "File Retrieval Failed (Check URLs or Paths)"
        }

    try:
        # 2. Parallel Execution Setup
        loop = asyncio.get_event_loop()
        
        # --- TASK A: Face Check (CPU/GPU Bound) ---
        # Compares Selfie <-> Front Card
        face_task = loop.run_in_executor(
            None, 
            face_agent.compare, 
            files['selfie'], 
            files['front']
        )

        # --- TASK B: Doc & Entity Pipeline ---
        async def run_doc_pipeline():
            # B1: Document Verification
            # Checks if Front/Back are valid Aadhaar cards & Gets Coordinates
            doc_res = await loop.run_in_executor(
                None, 
                doc_agent.verify_documents, 
                files['front'], 
                files['back']
            )
            
            # Fail Fast: If detection fails, stop pipeline here
            if not doc_res['success']:
                return {"success": False, "reason": doc_res['message']}

            # B2: Entity Extraction
            # Uses the crop coordinates from DocAgent to extract text immediately
            front_coords = doc_res.get('front_coords')
            entity_res = await loop.run_in_executor(
                None, 
                entity_agent.extract_from_file, 
                files['front'], 
                front_coords
            )
            
            # Return extracted data or empty dict
            return {
                "success": True, 
                "data": entity_res.get('data', {})
            }

        # Execute both branches concurrently
        face_score, doc_pipeline_res = await asyncio.gather(face_task, run_doc_pipeline())

        # 3. Aggregation Logic
        
        # If Document Pipeline failed (e.g., card not found), Reject immediately
        if not doc_pipeline_res['success']:
            return {
                "user_id": user_id,
                "status": "REJECTED",
                "score": 0,
                "reason": doc_pipeline_res.get('reason', 'Document Verification Failed')
            }

        # 4. Final Scoring
        entity_data = doc_pipeline_res['data']
        
        # Normalize Input Gender for comparison
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        # Calculate Score
        final_result = scorer.calculate_score(
            face_data={"similarity": face_score},
            entity_data=entity_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob
        )

        # 5. Construct Final Response
        # Map status to status code
        status_code_map = {
            "APPROVED": 2,
            "REJECTED": 1,
            "IN_REVIEW": 0
        }
        status_code = status_code_map.get(final_result['status'], 0)
        
        return {
            "user_id": user_id,
            "final_decision": final_result['status'],
            "status_code": status_code,
            "score": final_result['total_score'],
            "breakdown": final_result['breakdown'],
            "extracted_data": {
                "aadhaar": entity_data.get('aadharnumber'),
                "dob": entity_data.get('dob'),
                "gender": entity_data.get('gender')
            },
            "input_data": {
                "dob": req.dob,
                "gender": req.gender
            },
            "rejection_reasons": final_result['rejection_reasons']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "user_id": user_id,
            "status": "ERROR", 
            "message": str(e)
        }

    finally:
        # 6. Cleanup
        # Always delete the temp folder for this user after processing
        if task_dir and task_dir.exists():
            try:
                shutil.rmtree(task_dir)
                print(f"[{user_id}] Cleanup Complete")
            except Exception as e:
                print(f"[{user_id}] Cleanup Failed: {e}")

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production endpoint - returns only essential verification data"""
    user_id = str(req.user_id)
    print(f"[{user_id}] New Production Verification Request")

    # 1. Setup Files (Download/Copy)
    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "final_decision": "REJECTED",
            "status_code": 1,
            "extracted_data": {
                "aadhaar": None,
                "dob": None,
                "gender": None
            }
        }

    try:
        # 2. Parallel Execution Setup
        loop = asyncio.get_event_loop()
        
        # --- TASK A: Face Check (CPU/GPU Bound) ---
        face_task = loop.run_in_executor(
            None, 
            face_agent.compare, 
            files['selfie'], 
            files['front']
        )

        # --- TASK B: Doc & Entity Pipeline ---
        async def run_doc_pipeline():
            # B1: Document Verification
            doc_res = await loop.run_in_executor(
                None, 
                doc_agent.verify_documents, 
                files['front'], 
                files['back']
            )
            
            # Fail Fast: If detection fails, stop pipeline here
            if not doc_res['success']:
                return {"success": False, "reason": doc_res['message']}

            # B2: Entity Extraction
            front_coords = doc_res.get('front_coords')
            entity_res = await loop.run_in_executor(
                None, 
                entity_agent.extract_from_file, 
                files['front'], 
                front_coords
            )
            
            return {
                "success": True, 
                "data": entity_res.get('data', {})
            }

        # Execute both branches concurrently
        face_score, doc_pipeline_res = await asyncio.gather(face_task, run_doc_pipeline())

        # 3. Aggregation Logic
        if not doc_pipeline_res['success']:
            return {
                "user_id": user_id,
                "final_decision": "REJECTED",
                "status_code": 1,
                "extracted_data": {
                    "aadhaar": None,
                    "dob": None,
                    "gender": None
                }
            }

        # 4. Final Scoring
        entity_data = doc_pipeline_res['data']
        
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        final_result = scorer.calculate_score(
            face_data={"similarity": face_score},
            entity_data=entity_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob
        )

        # 5. Construct Production Response
        status_code_map = {
            "APPROVED": 2,
            "REJECTED": 1,
            "REVIEW": 0
        }
        status_code = status_code_map.get(final_result['status'], 0)
        
        return {
            "user_id": user_id,
            "final_decision": final_result['status'],
            "status_code": status_code,
            "extracted_data": {
                "aadhaar": entity_data.get('aadharnumber'),
                "dob": entity_data.get('dob'),
                "gender": entity_data.get('gender')
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "user_id": user_id,
            "final_decision": "REJECTED",
            "status_code": 1,
            "extracted_data": {
                "aadhaar": None,
                "dob": None,
                "gender": None
            }
        }

    finally:
        # 6. Cleanup
        if task_dir and task_dir.exists():
            try:
                shutil.rmtree(task_dir)
                print(f"[{user_id}] Cleanup Complete")
            except Exception as e:
                print(f"[{user_id}] Cleanup Failed: {e}")

if __name__ == "__main__":
    import uvicorn
    # Workers set to 1 to manage heavy model memory usage
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)