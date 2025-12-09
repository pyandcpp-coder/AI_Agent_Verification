import os
import shutil
import uuid
import asyncio
import aiohttp
import aiofiles
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# --- Import your actual Agents ---
# Ensure these files are in the same directory or properly installed as modules
from face_sim import FaceAgent
from fb_detect import DocAgent
from entity_agent import EntityAgent
from scoring import VerificationScorer

app = FastAPI()

# --- Global State ---
# We initialize agents here so models stay loaded in memory (RAM)
face_agent = None
doc_agent = None
entity_agent = None
scorer = None

TEMP_DIR = Path("temp")

@app.on_event("startup")
async def startup_event():
    global face_agent, doc_agent, entity_agent, scorer
    
    # Create temp dir if not exists
    TEMP_DIR.mkdir(exist_ok=True)

    print("--- STARTING SYSTEM INITIALIZATION ---")
    face_agent = FaceAgent()
    doc_agent = DocAgent()
    entity_agent = EntityAgent()
    scorer = VerificationScorer()
    print("--- SYSTEM READY ---")

class VerifyRequest(BaseModel):
    selfie_url: str
    front_url: str
    back_url: str
    gender: str = None  # Optional expected gender

async def download_single(session, url, save_path):
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                async with aiofiles.open(save_path, 'wb') as f:
                    await f.write(await resp.read())
                return True
    except Exception as e:
        print(f"Download Error {url}: {e}")
    return False

async def setup_files(task_id, req: VerifyRequest):
    """
    Downloads all 3 images to temp/{task_id}/ concurrently.
    Returns: Dict of paths or None if failed.
    """
    task_dir = TEMP_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "selfie": task_dir / "selfie.jpg",
        "front": task_dir / "front.jpg",
        "back": task_dir / "back.jpg"
    }

    async with aiohttp.ClientSession() as session:
        # Launch 3 downloads at once
        results = await asyncio.gather(
            download_single(session, req.selfie_url, paths["selfie"]),
            download_single(session, req.front_url, paths["front"]),
            download_single(session, req.back_url, paths["back"])
        )

    # Check if all downloads succeeded
    if all(results):
        return {k: str(v) for k, v in paths.items()} # Return string paths
    else:
        # Cleanup immediately if download failed
        shutil.rmtree(task_dir)
        return None

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    task_id = str(uuid.uuid4())
    print(f"[{task_id}] New Verification Request")

    # 1. Download Files
    files = await setup_files(task_id, req)
    if not files:
        return {"status": "FAILED", "reason": "Image Download Failed"}

    try:
        # 2. Parallel Execution (Face vs Doc+Entity)
        
        # Task A: Face Comparison (Selfie <-> Front Card)
        # Note: FaceAgent is synchronous (CPU/GPU bound), so we run it in a thread
        loop = asyncio.get_event_loop()
        face_task = loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])

        # Task B: Document Pipeline
        async def run_doc_pipeline():
            # B1: Doc Check
            # DocAgent is synchronous, run in thread
            doc_res = await loop.run_in_executor(None, doc_agent.verify_documents, files['front'], files['back'])
            
            if not doc_res['success']:
                return {"success": False, "reason": doc_res['message']}

            # B2: Entity Extraction (Only if Doc Check Passed)
            # We pass the coords so EntityAgent can crop instantly
            front_coords = doc_res.get('front_coords')
            
            # EntityAgent is synchronous, run in thread
            entity_res = await loop.run_in_executor(
                None, 
                entity_agent.extract_from_file, 
                files['front'], 
                front_coords
            )
            
            return {"success": True, "data": entity_res.get('data', {}), "raw_entity_res": entity_res}

        # Wait for both branches
        face_score, doc_pipeline_res = await asyncio.gather(face_task, run_doc_pipeline())

        # 3. Aggregation Logic
        
        # Check if Pipeline failed (e.g. invalid docs)
        if not doc_pipeline_res['success']:
            return {
                "status": "REJECTED",
                "score": 0,
                "reason": doc_pipeline_res.get('reason', 'Document Verification Failed')
            }

        # 4. Scoring
        entity_data = doc_pipeline_res['data']
        
        # Prepare inputs for scorer
        # scorer expects: face_data dict, entity_data dict, expected_gender string
        final_result = scorer.calculate_score(
            face_data={"similarity": face_score},
            entity_data=entity_data,
            expected_gender=req.gender
        )

        return {
            "task_id": task_id,
            "final_decision": final_result['status'],
            "score": final_result['total_score'],
            "breakdown": final_result['breakdown'],
            "extracted_data": {
                "aadhaar": entity_data.get('aadharnumber'),
                "dob": entity_data.get('dob'),
                "gender": entity_data.get('gender')
            },
            "rejection_reasons": final_result['rejection_reasons']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "message": str(e)}

    finally:
        # 5. Cleanup
        # Always delete the temp folder for this task
        task_dir = TEMP_DIR / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
            print(f"[{task_id}] Cleanup Complete")

if __name__ == "__main__":
    import uvicorn
    # Workers=1 because our Agents hold heavy models in memory. 
    # Multiple workers would duplicate models and crash RAM.
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)