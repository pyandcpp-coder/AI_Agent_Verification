import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from pathlib import Path


BASE_URL = "https://qoneqt.com/v1/api"
ADMIN_ID = 27  # Your Main Admin ID
AGENT_ID = 77 # The Specific Agent ID you are running (Change this for other instances: 77, 78, 79, 80)
BATCH_SIZE = 20
TTL_HOURS = 12



LOCAL_AI_URL = "http://localhost:8100/verification/verify/agent/"

# Create logs directory structure
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Browser-like Headers to bypass Cloudflare
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://qoneqt.com",
    "Referer": "https://qoneqt.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"'
}

# Session tracking for batch runs
current_session = {
    "session_id": None,
    "start_time": None,
    "batch_number": 0,
    "users_processed": []
}

def create_session_log_file():
    """Create a new session log file for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session["session_id"] = f"agent_{AGENT_ID}_{timestamp}"
    current_session["start_time"] = datetime.now().isoformat()
    current_session["batch_number"] = 0
    current_session["users_processed"] = []
    
    # Create agent-specific directory
    agent_dir = LOGS_DIR / f"agent_{AGENT_ID}"
    agent_dir.mkdir(exist_ok=True)
    
    # Create session directory
    session_dir = agent_dir / current_session["session_id"]
    session_dir.mkdir(exist_ok=True)
    
    return session_dir

def save_user_log(user_id, log_data, session_dir):
    """Save detailed log for a single user"""
    user_log_file = session_dir / f"user_{user_id}.json"
    
    with open(user_log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved detailed log for User {user_id}")

def save_batch_summary(session_dir, batch_data):
    """Save summary of the current batch"""
    current_session["batch_number"] += 1
    batch_file = session_dir / f"batch_{current_session['batch_number']}_summary.json"
    
    summary = {
        "batch_number": current_session["batch_number"],
        "timestamp": datetime.now().isoformat(),
        "agent_id": AGENT_ID,
        "total_users": len(batch_data),
        "users": batch_data
    }
    
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved batch summary: {batch_file.name}")

def save_session_summary(session_dir):
    """Save overall session summary at the end"""
    session_file = session_dir / "session_summary.json"
    
    summary = {
        "session_id": current_session["session_id"],
        "agent_id": AGENT_ID,
        "start_time": current_session["start_time"],
        "end_time": datetime.now().isoformat(),
        "total_batches": current_session["batch_number"],
        "total_users_processed": len(current_session["users_processed"]),
        "user_ids": current_session["users_processed"],
        "statistics": calculate_session_statistics(session_dir)
    }
    
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved session summary: {session_file.name}")

def calculate_session_statistics(session_dir):
    """Calculate statistics from all user logs in the session"""
    stats = {
        "approved": 0,
        "rejected": 0,
        "review": 0,
        "failed": 0,
        "total_processing_time": 0
    }
    
    # Read all user log files
    for user_file in session_dir.glob("user_*.json"):
        try:
            with open(user_file, 'r') as f:
                user_data = json.load(f)
                
            decision = user_data.get("final_result", {}).get("final_decision", "FAILED")
            if decision == "APPROVED":
                stats["approved"] += 1
            elif decision == "REJECTED":
                stats["rejected"] += 1
            elif decision == "REVIEW":
                stats["review"] += 1
            else:
                stats["failed"] += 1
            
            stats["total_processing_time"] += user_data.get("timing", {}).get("total_time", 0)
        except:
            pass
    
    if stats["approved"] + stats["rejected"] + stats["review"] + stats["failed"] > 0:
        stats["avg_processing_time"] = stats["total_processing_time"] / (stats["approved"] + stats["rejected"] + stats["review"] + stats["failed"])
    else:
        stats["avg_processing_time"] = 0
    
    return stats

async def lock_batch(session):
    """Step 1: Lock a batch of users for this agent"""
    url = f"{BASE_URL}/admin/kyc-lock-batch"
    payload = {
        "admin_id": ADMIN_ID,
        "target_admin_id": AGENT_ID,
        "batch_size": BATCH_SIZE,
        "ttl_hours": TTL_HOURS
    }
    
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("success"):
                    logger.info(f"🔒 Locked {data.get('locked_count')} users for Agent {AGENT_ID}")
                    return data.get("kyc_ids", [])
                else:
                    logger.error(f"Failed to lock batch: {data.get('message')}")
            else:
                logger.error(f"API Error {resp.status}: {await resp.text()}")
    except Exception as e:
        logger.error(f"Lock Batch Exception: {e}")
    return []

async def fetch_user_details(session):
    """Step 2: Get details for the assigned users"""
    url = f"{BASE_URL}/admin/verify-list"
    # Note: 'admin_id' here acts as the filter for the assigned agent
    payload = {
        "admin_id": AGENT_ID, 
        "page": 1,
        "limit": BATCH_SIZE,
        "statusFilter": "2",  # Pending?
        "isFullSearch": True,
        "offset": 0,
        "type": "0",
        "assignment_filter": "assigned"
    }
    
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status == 200:
                data = await resp.json()
                users = data.get("data", [])
                logger.info(f"📥 Fetched details for {len(users)} users")
                return users
            else:
                logger.error(f"Fetch Details Error {resp.status}: {await resp.text()}")
    except Exception as e:
        logger.error(f"Fetch Details Exception: {e}")
    return []

async def process_single_user(session, user, session_dir):
    """Step 3 & 4: Process via Local AI and Push Result"""
    user_id = user.get("user_id")
    
    # Initialize comprehensive log for this user
    user_log = {
        "user_id": user_id,
        "agent_id": AGENT_ID,
        "session_id": current_session["session_id"],
        "batch_number": current_session["batch_number"] + 1,
        "timestamp": datetime.now().isoformat(),
        "input_data": {},
        "processing_steps": [],
        "timing": {},
        "api_calls": [],
        "final_result": {},
        "errors": []
    }
    
    start_time = time.time()
    
    try:
        # Store original input data
        user_log["input_data"] = {
            "user_id": user_id,
            "dob": user.get("dob"),
            "gender": user.get("gender"),
            "passport_first": user.get("passport_first"),
            "passport_old": user.get("passport_old"),
            "selfie_photo": user.get("selfie_photo"),
            "full_user_data": user  # Store complete user object
        }
        
        user_log["processing_steps"].append({
            "step": "1_input_received",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "User data received from main server"
        })
        
        # 1. Prepare Request for Local AI
        ai_request = {
            "user_id": user_id,
            "dob": user.get("dob"),
            "passport_first": user.get("passport_first"),
            "passport_old": user.get("passport_old"),
            "selfie_photo": user.get("selfie_photo"),
            "gender": user.get("gender")
        }
        
        # Handle image paths
        CDN_BASE = "https://cdn.qoneqt.com/" 
        for key in ["passport_first", "passport_old", "selfie_photo"]:
            if ai_request[key] and not ai_request[key].startswith("http"):
                ai_request[key] = CDN_BASE + ai_request[key]
        
        user_log["processing_steps"].append({
            "step": "2_image_paths_resolved",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Image URLs prepared",
            "resolved_urls": {
                "passport_first": ai_request["passport_first"],
                "passport_old": ai_request["passport_old"],
                "selfie_photo": ai_request["selfie_photo"]
            }
        })

        # 2. Call Local AI
        ai_start_time = time.time()
        try:
            async with session.post(LOCAL_AI_URL, json=ai_request, timeout=120) as resp:
                ai_elapsed = time.time() - ai_start_time
                
                api_call_log = {
                    "endpoint": LOCAL_AI_URL,
                    "method": "POST",
                    "request": ai_request,
                    "status_code": resp.status,
                    "elapsed_time": ai_elapsed,
                    "timestamp": datetime.now().isoformat()
                }
                
                if resp.status != 200:
                    error_text = await resp.text()
                    api_call_log["error"] = error_text
                    user_log["api_calls"].append(api_call_log)
                    user_log["errors"].append(f"Local AI Failed: Status {resp.status}")
                    user_log["processing_steps"].append({
                        "step": "3_local_ai_processing",
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed",
                        "message": f"AI processing failed with status {resp.status}",
                        "error": error_text
                    })
                    logger.error(f"❌ User {user_id}: Local AI Failed ({resp.status})")
                    user_log["final_result"] = {
                        "final_decision": "FAILED",
                        "status_code": -1,
                        "reason": "Local AI processing failed",
                        "rejection_reasons": [f"Local AI processing failed with status {resp.status}"],
                        "extracted_data": {}
                    }
                    save_user_log(user_id, user_log, session_dir)
                    return user_log
                
                ai_result = await resp.json()
                api_call_log["response"] = ai_result
                user_log["api_calls"].append(api_call_log)
                
                user_log["processing_steps"].append({
                    "step": "3_local_ai_processing",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "message": "AI processing completed successfully",
                    "processing_time": ai_elapsed,
                    "ai_decision": ai_result.get("final_decision"),
                    "ai_rejection_reasons": ai_result.get("rejection_reasons", []),
                    "ai_result": ai_result
                })
                
        except asyncio.TimeoutError:
            ai_elapsed = time.time() - ai_start_time
            user_log["errors"].append("Local AI timeout after 120 seconds")
            user_log["processing_steps"].append({
                "step": "3_local_ai_processing",
                "timestamp": datetime.now().isoformat(),
                "status": "timeout",
                "message": "AI processing timed out",
                "elapsed_time": ai_elapsed
            })
            logger.error(f"❌ User {user_id}: Local AI Timeout")
            user_log["final_result"] = {
                "final_decision": "FAILED",
                "status_code": -1,
                "reason": "Local AI timeout",
                "rejection_reasons": ["AI processing timeout after 120 seconds"],
                "extracted_data": {}
            }
            save_user_log(user_id, user_log, session_dir)
            return user_log
            
        # 3. Push Result to Main Server
        push_url = f"{BASE_URL}/ai-agent"
        push_payload = {
            "user_id": user_id,
            "agent_id": str(AGENT_ID),
            "final_decision": ai_result.get("final_decision", "REJECTED"),
            "status_code": ai_result.get("status_code", 0),
            "extracted_data": ai_result.get("extracted_data", {}),
            "rejection_reasons": ai_result.get("rejection_reasons", [])
        }
        
        push_start_time = time.time()
        try:
            async with session.post(push_url, json=push_payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as push_resp:
                push_elapsed = time.time() - push_start_time
                
                push_call_log = {
                    "endpoint": push_url,
                    "method": "POST",
                    "request": push_payload,
                    "status_code": push_resp.status,
                    "elapsed_time": push_elapsed,
                    "timestamp": datetime.now().isoformat()
                }
                
                if push_resp.status == 200:
                    push_result = await push_resp.json()
                    push_call_log["response"] = push_result
                    user_log["api_calls"].append(push_call_log)
                    
                    user_log["processing_steps"].append({
                        "step": "4_result_pushed",
                        "timestamp": datetime.now().isoformat(),
                        "status": "success",
                        "message": "Result successfully pushed to main server",
                        "push_time": push_elapsed,
                        "server_response": push_result
                    })
                    
                    # Log with rejection reasons if available
                    decision_log = f"{push_payload['final_decision']}"
                    if push_payload['rejection_reasons']:
                        decision_log += f" - Reasons: {', '.join(push_payload['rejection_reasons'])}"
                    logger.info(f"✅ User {user_id}: Processed & Pushed ({decision_log})")
                else:
                    error_text = await push_resp.text()
                    push_call_log["error"] = error_text
                    user_log["api_calls"].append(push_call_log)
                    user_log["errors"].append(f"Push failed: Status {push_resp.status}")
                    
                    user_log["processing_steps"].append({
                        "step": "4_result_pushed",
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed",
                        "message": f"Failed to push result to main server",
                        "status_code": push_resp.status,
                        "error": error_text
                    })
                    
                    logger.error(f"⚠️ User {user_id}: Failed to Push Result {push_resp.status} - {error_text[:100]}")
                    
        except Exception as push_error:
            push_elapsed = time.time() - push_start_time
            user_log["errors"].append(f"Push exception: {str(push_error)}")
            user_log["processing_steps"].append({
                "step": "4_result_pushed",
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": "Exception during result push",
                "error": str(push_error),
                "elapsed_time": push_elapsed
            })
            logger.error(f"🔥 User {user_id}: Push Exception - {push_error}")
        
        # Store final result
        user_log["final_result"] = {
            "final_decision": ai_result.get("final_decision", "REJECTED"),
            "status_code": ai_result.get("status_code", 0),
            "extracted_data": ai_result.get("extracted_data", {}),
            "rejection_reasons": ai_result.get("rejection_reasons", [])
        }

    except Exception as e:
        user_log["errors"].append(f"Critical exception: {str(e)}")
        user_log["processing_steps"].append({
            "step": "error",
            "timestamp": datetime.now().isoformat(),
            "status": "critical_error",
            "message": "Unhandled exception during processing",
            "error": str(e)
        })
        user_log["final_result"] = {
            "final_decision": "FAILED",
            "status_code": -1,
            "reason": f"Exception: {str(e)}",
            "rejection_reasons": [f"Critical exception: {str(e)}"],
            "extracted_data": {}
        }
        logger.error(f"🔥 User {user_id} Exception: {e}")
    
    # Calculate timing
    total_time = time.time() - start_time
    user_log["timing"] = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "total_time": total_time,
        "ai_processing_time": sum(
            step.get("processing_time", 0) 
            for step in user_log["processing_steps"] 
            if "processing_time" in step
        ),
        "api_push_time": sum(
            step.get("push_time", 0) 
            for step in user_log["processing_steps"] 
            if "push_time" in step
        )
    }
    
    # Save the comprehensive log
    save_user_log(user_id, user_log, session_dir)
    current_session["users_processed"].append(user_id)
    
    return user_log

async def run_pipeline():
    # Create session log directory
    session_dir = create_session_log_file()
    logger.info(f"📁 Session logs will be saved to: {session_dir}")
    
    async with aiohttp.ClientSession() as session:
        try:
            while True:
                logger.info(f"--- Starting Cycle for Agent {AGENT_ID} ---")
                
                # 1. Lock Batch
                locked_ids = await lock_batch(session)
                
                if not locked_ids:
                    logger.info("No users available to lock. Sleeping for 60s...")
                    await asyncio.sleep(60)
                    continue
                
                # 2. Fetch User Details
                users_to_process = await fetch_user_details(session)
                
                if not users_to_process:
                    logger.warning("Locked batch but got no user details? Retrying in 10s...")
                    await asyncio.sleep(10)
                    continue
                
                # 3. Process Batch
                batch_start_time = time.time()
                batch_results = []
                
                # Process in chunks of 5 users at a time
                chunk_size = 5
                for i in range(0, len(users_to_process), chunk_size):
                    chunk = users_to_process[i:i + chunk_size]
                    tasks = [process_single_user(session, u, session_dir) for u in chunk]
                    chunk_results = await asyncio.gather(*tasks)
                    batch_results.extend(chunk_results)
                
                batch_elapsed = time.time() - batch_start_time
                
                # Create batch summary
                batch_summary = []
                for user_log in batch_results:
                    if user_log:
                        batch_summary.append({
                            "user_id": user_log["user_id"],
                            "final_decision": user_log["final_result"].get("final_decision"),
                            "status_code": user_log["final_result"].get("status_code"),
                            "rejection_reasons": user_log["final_result"].get("rejection_reasons", []),
                            "extracted_data": user_log["final_result"].get("extracted_data", {}),
                            "processing_time": user_log["timing"].get("total_time"),
                            "errors": len(user_log["errors"])
                        })
                
                # Save batch summary
                save_batch_summary(session_dir, {
                    "batch_elapsed_time": batch_elapsed,
                    "users": batch_summary
                })
                
                logger.info(f"--- Batch Complete. Processed {len(batch_results)} users in {batch_elapsed:.2f}s ---")
                
                # Optional sleep to be nice to the server
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested...")
            raise
        finally:
            # Save session summary before exiting
            logger.info("💾 Saving final session summary...")
            save_session_summary(session_dir)
            logger.info(f"✅ All logs saved to: {session_dir}")

if __name__ == "__main__":
    print(f"🚀 Starting Production Dispatcher for Agent {AGENT_ID}")
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("Stopping...")