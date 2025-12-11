import os
import sys
import shutil
import asyncio
import aiohttp
import aiofiles
import cloudscraper
import gc
import weakref
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union
from contextlib import asynccontextmanager

# Ensure the current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- IMPORT ACTUAL AGENTS ---
try:
    from app.face_sim import FaceAgent
    from app.fb_detect import DocAgent
    from app.entity import EntityAgent
    from app.gender_pipeline import GenderPipeline
    from scoring import VerificationScorer
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"Current directory: {current_dir}")
    print(f"sys.path: {sys.path}")
    raise

# --- Global State ---
face_agent = None
doc_agent = None
entity_agent = None
gender_pipeline = None
scorer = None
http_session = None

TEMP_DIR = Path("temp")

# Track active sessions with weak references to prevent memory leaks
active_sessions = weakref.WeakSet()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global face_agent, doc_agent, entity_agent, gender_pipeline, scorer, http_session
    
    # Startup
    TEMP_DIR.mkdir(exist_ok=True)
    print("--- STARTING SYSTEM INITIALIZATION ---")
    
    # Create persistent HTTP session with stricter connection limits
    connector = aiohttp.TCPConnector(
        limit=50,  # Reduced from 100
        limit_per_host=10,  # Reduced from 30
        ttl_dns_cache=300,
        force_close=True,
        enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    http_session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    active_sessions.add(http_session)
    
    face_agent = FaceAgent()
    doc_agent = DocAgent()
    entity_agent = EntityAgent()
    gender_pipeline = GenderPipeline()
    scorer = VerificationScorer()
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    print("--- SYSTEM READY ---")
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    if http_session:
        await http_session.close()
        await asyncio.sleep(0.25)  # Give time for connections to close
    
    # Final cleanup of temp directory
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except Exception as e:
            print(f"Final cleanup error: {e}")
    
    print("--- SYSTEM SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

async def periodic_cleanup():
    """Background task to clean up old temp directories every 2 minutes."""
    import time
    while True:
        try:
            await asyncio.sleep(120)  # Run every 2 minutes (reduced from 5)
            
            if not TEMP_DIR.exists():
                continue
            
            current_time = time.time()
            cleaned_count = 0
            
            # Delete directories older than 5 minutes (reduced from 10)
            for user_dir in TEMP_DIR.iterdir():
                if not user_dir.is_dir():
                    continue
                
                dir_age = current_time - user_dir.stat().st_mtime
                if dir_age > 300:  # 5 minutes
                    try:
                        force_cleanup_directory(user_dir)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"Periodic cleanup failed for {user_dir}: {e}")
            
            if cleaned_count > 0:
                print(f"Periodic cleanup: Removed {cleaned_count} old temp directories")
                gc.collect()  # Force garbage collection after cleanup
                
        except Exception as e:
            print(f"Error in periodic cleanup: {e}")

def force_cleanup_directory(directory: Path, max_retries: int = 3):
    """Aggressively cleanup a directory with retries."""
    for attempt in range(max_retries):
        try:
            # Force garbage collection to close file handles
            gc.collect()
            
            # Try to close any open file descriptors in this directory
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                # Don't check all FDs, just do aggressive cleanup
            except:
                pass
            
            # First try: normal removal
            if attempt == 0:
                shutil.rmtree(directory, ignore_errors=False)
                return
            
            # Second try: ignore errors
            if attempt == 1:
                shutil.rmtree(directory, ignore_errors=True)
                if not directory.exists():
                    return
            
            # Final try: manual file-by-file deletion
            if directory.exists():
                for item in directory.rglob('*'):
                    try:
                        if item.is_file():
                            item.unlink(missing_ok=True)
                        elif item.is_dir():
                            try:
                                item.rmdir()
                            except:
                                pass
                    except Exception:
                        pass
                
                try:
                    directory.rmdir()
                except:
                    pass
            
            if not directory.exists():
                return
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to cleanup {directory} after {max_retries} attempts: {e}")
            else:
                import time
                time.sleep(0.1 * (attempt + 1))  # Progressive backoff

# --- INPUT MODEL ---
class VerifyRequest(BaseModel):
    user_id: Union[int, str]
    dob: Optional[str] = None
    passport_first: str
    passport_old: str
    selfie_photo: str
    gender: Optional[str] = None

async def fetch_file(session: aiohttp.ClientSession, source: str, destination: Path) -> bool:
    """
    Hybrid Fetcher with proper resource management.
    """
    try:
        if str(source).startswith(('http://', 'https://')):
            loop = asyncio.get_event_loop()
            
            def download_with_cloudscraper(url: str) -> bytes:
                scraper = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'darwin',
                        'mobile': False
                    }
                )
                try:
                    response = scraper.get(url, timeout=30)
                    response.raise_for_status()
                    return response.content
                finally:
                    # Ensure scraper session is closed
                    if hasattr(scraper, 'close'):
                        scraper.close()
            
            content = await loop.run_in_executor(None, download_with_cloudscraper, str(source))
            
            # Write content to file with explicit close
            async with aiofiles.open(destination, 'wb') as f:
                await f.write(content)
            
            print(f"Downloaded: {source} ({len(content)} bytes)")
            return True
            
        else:
            source_path = Path(source)
            if source_path.exists():
                # Use copy2 to preserve metadata
                shutil.copy2(source_path, destination)
                return True
            else:
                print(f"Local file not found: {source}")
    except Exception as e:
        print(f"Error fetching {source}: {e}")
    return False

async def setup_files(user_id: str, req: VerifyRequest):
    """Sets up temp environment with aggressive cleanup."""
    task_dir = TEMP_DIR / str(user_id)
    
    # Clean existing directory if it exists
    if task_dir.exists():
        force_cleanup_directory(task_dir)
    
    task_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "selfie": task_dir / f"{user_id}_selfie.jpg",
        "front": task_dir / f"{user_id}_aadhar_front.jpg",
        "back": task_dir / f"{user_id}_aadhar_back.jpg"
    }

    global http_session
    results = await asyncio.gather(
        fetch_file(http_session, req.selfie_photo, paths["selfie"]),
        fetch_file(http_session, req.passport_first, paths["front"]),
        fetch_file(http_session, req.passport_old, paths["back"]),
        return_exceptions=True
    )

    # Check results
    if all(r is True for r in results):
        return {k: str(v) for k, v in paths.items()}, task_dir
    else:
        force_cleanup_directory(task_dir)
        return None, None

@app.post("/verification/verify")
async def verify_user(req: VerifyRequest):
    """Full verification endpoint with detailed breakdown."""
    user_id = str(req.user_id)
    print(f"[{user_id}] New Verification Request")

    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "status": "FAILED", 
            "reason": "File Retrieval Failed"
        }

    try:
        loop = asyncio.get_event_loop()
        
        face_task = loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])

        async def run_doc_pipeline():
            doc_res = await loop.run_in_executor(None, doc_agent.verify_documents, files['front'], files['back'])
            
            if not doc_res['success']:
                return {"success": False, "reason": doc_res['message'], "data": {}}

            front_coords = doc_res.get('front_coords')
            entity_task = loop.run_in_executor(None, entity_agent.extract_from_file, files['front'], front_coords)
            gender_task = loop.run_in_executor(None, gender_pipeline.detect_gender, files['front'])
            
            entity_res, gender_res = await asyncio.gather(entity_task, gender_task)
            
            merged_data = entity_res.get('data', {})
            ocr_gender = merged_data.get('gender', 'Other')
            
            if ocr_gender in ['Other', 'Not Detected', None] and gender_res.face_detected:
                detected_gender = gender_res.gender.capitalize()
                if detected_gender in ['Male', 'Female']:
                    print(f"[{user_id}] OCR Gender '{ocr_gender}' -> Fallback: {detected_gender}")
                    merged_data['gender'] = detected_gender
                    merged_data['gender_source'] = 'pipeline'
                else:
                    merged_data['gender_source'] = 'ocr_failed'
            else:
                merged_data['gender_source'] = 'ocr'
            
            return {"success": True, "data": merged_data}

        face_score, doc_pipeline_res = await asyncio.gather(face_task, run_doc_pipeline())

        if not doc_pipeline_res['success']:
            return {
                "user_id": user_id,
                "status": "REJECTED",
                "score": 0,
                "reason": doc_pipeline_res.get('reason', 'Document Verification Failed')
            }

        entity_data = doc_pipeline_res['data']
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        final_result = scorer.calculate_score(
            face_data=face_score,
            entity_data=entity_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob
        )

        status_code_map = {"APPROVED": 2, "REJECTED": 1, "REVIEW": 0}
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
        # Immediate cleanup
        if task_dir and task_dir.exists():
            force_cleanup_directory(task_dir)
            print(f"[{user_id}] Cleanup Complete")
        
        # Force garbage collection
        gc.collect()

@app.post("/verification/verify/agent/")
async def verify_user_production(req: VerifyRequest):
    """Production endpoint - essential data only."""
    user_id = str(req.user_id)
    print(f"[{user_id}] Production Verification Request")

    files, task_dir = await setup_files(user_id, req)
    if not files:
        return {
            "user_id": user_id,
            "final_decision": "REJECTED",
            "status_code": 1,
            "extracted_data": {"aadhaar": None, "dob": None, "gender": None}
        }

    try:
        loop = asyncio.get_event_loop()
        face_task = loop.run_in_executor(None, face_agent.compare, files['selfie'], files['front'])

        async def run_doc_pipeline():
            doc_res = await loop.run_in_executor(None, doc_agent.verify_documents, files['front'], files['back'])
            
            if not doc_res['success']:
                return {"success": False, "reason": doc_res['message']}

            front_coords = doc_res.get('front_coords')
            entity_task = loop.run_in_executor(None, entity_agent.extract_from_file, files['front'], front_coords)
            gender_task = loop.run_in_executor(None, gender_pipeline.detect_gender, files['front'])
            
            entity_res, gender_res = await asyncio.gather(entity_task, gender_task)
            
            merged_data = entity_res.get('data', {})
            ocr_gender = merged_data.get('gender', 'Other')
            
            if ocr_gender in ['Other', 'Not Detected', None] and gender_res.face_detected:
                detected_gender = gender_res.gender.capitalize()
                if detected_gender in ['Male', 'Female']:
                    merged_data['gender'] = detected_gender
                    merged_data['gender_source'] = 'pipeline'
                else:
                    merged_data['gender_source'] = 'ocr_failed'
            else:
                merged_data['gender_source'] = 'ocr'
            
            return {"success": True, "data": merged_data}

        face_score, doc_pipeline_res = await asyncio.gather(face_task, run_doc_pipeline())

        if not doc_pipeline_res['success']:
            return {
                "user_id": user_id,
                "final_decision": "REJECTED",
                "status_code": 1,
                "extracted_data": {"aadhaar": None, "dob": None, "gender": None}
            }

        entity_data = doc_pipeline_res['data']
        expected_gender = req.gender.lower() if req.gender else None
        expected_dob = req.dob if req.dob else None
        
        final_result = scorer.calculate_score(
            face_data=face_score,
            entity_data=entity_data,
            expected_gender=expected_gender,
            expected_dob=expected_dob
        )

        status_code_map = {"APPROVED": 2, "REJECTED": 1, "REVIEW": 0}
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
            "rejection_reasons": final_result['rejection_reasons']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "user_id": user_id,
            "final_decision": "REJECTED",
            "status_code": 1,
            "extracted_data": {"aadhaar": None, "dob": None, "gender": None}
        }

    finally:
        if task_dir and task_dir.exists():
            force_cleanup_directory(task_dir)
            print(f"[{user_id}] Cleanup Complete")
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)