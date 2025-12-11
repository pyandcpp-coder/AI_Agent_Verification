import asyncio
import aiohttp
import time
import json
import logging
from datetime import datetime
from pathlib import Path


BASE_URL = "https://qoneqt.com/v1/api"
ADMIN_ID = 27  
AGENT_IDS = [77, 78, 79, 80]  
BATCH_SIZE = 20
TTL_HOURS = 12

LOCAL_AI_URL = "http://localhost:8101/verification/verify/agent/"

# Retry configuration for 502 errors
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
AGENT_CYCLE_DELAY = 3  # delay between agents
FULL_CYCLE_DELAY = 10  # delay between full cycles

# Set to False to suppress detailed 502 HTML error logs
LOG_502_DETAILS = False

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("agent_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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


current_session = {
    "session_id": None,
    "start_time": None,
    "batch_number": 0,
    "users_processed": [],
    "agent_id": None
}

def create_session_log_file(agent_id):
    """Create a new session log file for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_session["session_id"] = f"agent_{agent_id}_{timestamp}"
    current_session["start_time"] = datetime.now().isoformat()
    current_session["batch_number"] = 0
    current_session["users_processed"] = []
    current_session["agent_id"] = agent_id
    
    # Create agent-specific directory
    agent_dir = LOGS_DIR / f"agent_{agent_id}"
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
        "agent_id": current_session["agent_id"],
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
        "agent_id": current_session["agent_id"],
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
    """Calculate detailed statistics from all user logs in the session"""
    stats = {
        "approved": 0,
        "rejected": 0,
        "review": 0,
        "failed": 0,
        "total_processing_time": 0,
        "total_score": 0,
        "rejection_reasons": [],
        "avg_score_approved": 0,
        "avg_score_rejected": 0,
        "breakdown_summary": {
            "face_similarity": [],
            "age_verification": [],
            "gender_match": [],
            "dob_match": []
        }
    }
    
    approved_scores = []
    rejected_scores = []
    
    # Read all user log files
    for user_file in session_dir.glob("user_*.json"):
        try:
            with open(user_file, 'r') as f:
                user_data = json.load(f)
                
            final_result = user_data.get("final_result", {})
            decision = final_result.get("final_decision", "FAILED")
            score = final_result.get("score", 0)
            breakdown = final_result.get("breakdown", {})
            
            if decision == "APPROVED":
                stats["approved"] += 1
                approved_scores.append(score)
            elif decision == "REJECTED":
                stats["rejected"] += 1
                rejected_scores.append(score)
                # Collect rejection reasons
                stats["rejection_reasons"].extend(final_result.get("rejection_reasons", []))
            elif decision == "REVIEW":
                stats["review"] += 1
            else:
                stats["failed"] += 1
            
            stats["total_processing_time"] += user_data.get("timing", {}).get("total_time", 0)
            stats["total_score"] += score
            
            # Collect breakdown data
            if breakdown:
                for key in stats["breakdown_summary"].keys():
                    if key in breakdown:
                        stats["breakdown_summary"][key].append(breakdown[key])
        except Exception as e:
            logger.error(f"Error processing stats for {user_file}: {e}")
            pass
    
    total_users = stats["approved"] + stats["rejected"] + stats["review"] + stats["failed"]
    
    if total_users > 0:
        stats["avg_processing_time"] = stats["total_processing_time"] / total_users
        stats["avg_score"] = stats["total_score"] / total_users
    else:
        stats["avg_processing_time"] = 0
        stats["avg_score"] = 0
    
    if approved_scores:
        stats["avg_score_approved"] = sum(approved_scores) / len(approved_scores)
    if rejected_scores:
        stats["avg_score_rejected"] = sum(rejected_scores) / len(rejected_scores)
    
    # Count rejection reasons
    from collections import Counter
    if stats["rejection_reasons"]:
        reason_counts = Counter(stats["rejection_reasons"])
        stats["top_rejection_reasons"] = dict(reason_counts.most_common(10))
    else:
        stats["top_rejection_reasons"] = {}
    
    # Calculate average breakdown scores
    for key, values in stats["breakdown_summary"].items():
        if values:
            stats["breakdown_summary"][key] = {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        else:
            stats["breakdown_summary"][key] = {"avg": 0, "min": 0, "max": 0, "count": 0}
    
    return stats

async def lock_batch(session, agent_id, retry_count=0):
    """Step 1: Lock a batch of users for this agent with retry logic"""
    url = f"{BASE_URL}/admin/kyc-lock-batch"
    payload = {
        "admin_id": ADMIN_ID,
        "target_admin_id": agent_id,
        "batch_size": BATCH_SIZE,
        "ttl_hours": TTL_HOURS
    }
    
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("success"):
                    logger.info(f"🔒 Locked {data.get('locked_count')} users for Agent {agent_id}")
                    return data.get("kyc_ids", [])
                else:
                    logger.error(f"Failed to lock batch: {data.get('message')}")
            elif resp.status == 502 and retry_count < MAX_RETRIES:
                # Server is down, implement exponential backoff
                wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY  # 5s, 10s, 20s
                logger.warning(f"⚠️ 502 Bad Gateway for Agent {agent_id}. Server may be down. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait_time)
                return await lock_batch(session, agent_id, retry_count + 1)
            else:
                error_text = await resp.text()
                # Only log detailed error if enabled
                if LOG_502_DETAILS or resp.status != 502:
                    logger.error(f"API Error {resp.status} for Agent {agent_id}: {error_text[:200]}")
                else:
                    logger.error(f"API Error {resp.status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout locking batch for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Lock Batch Exception for Agent {agent_id}: {e}")
    return []

async def release_batch(session, agent_id, retry_count=0):
    """Release the batch for an agent after processing with retry logic"""
    url = f"{BASE_URL}/admin/kyc-release-batch"
    payload = {
        "admin_id": ADMIN_ID,
        "target_admin_id": agent_id
    }
    
    try:
        async with session.post(url, json=payload, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("success"):
                    logger.info(f"🔓 Released batch for Agent {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to release batch for Agent {agent_id}: {data.get('message')}")
            elif resp.status == 502 and retry_count < MAX_RETRIES:
                wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY
                logger.warning(f"⚠️ 502 Bad Gateway releasing batch for Agent {agent_id}. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait_time)
                return await release_batch(session, agent_id, retry_count + 1)
            else:
                error_text = await resp.text()
                if LOG_502_DETAILS or resp.status != 502:
                    logger.error(f"Release Batch API Error {resp.status} for Agent {agent_id}: {error_text[:200]}")
                else:
                    logger.error(f"Release Batch API Error {resp.status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout releasing batch for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Release Batch Exception for Agent {agent_id}: {e}")
    return False

async def fetch_user_details(session, agent_id, retry_count=0):
    """Step 2: Get details for the assigned users with retry logic"""
    url = f"{BASE_URL}/admin/verify-list"
    payload = {
        "admin_id": agent_id, 
        "page": 1,
        "limit": BATCH_SIZE,
        "statusFilter": "2",  
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
                logger.info(f"📋 Fetched details for {len(users)} users (Agent {agent_id})")
                return users
            elif resp.status == 502 and retry_count < MAX_RETRIES:
                wait_time = 2 ** retry_count * INITIAL_RETRY_DELAY
                logger.warning(f"⚠️ 502 Bad Gateway fetching users for Agent {agent_id}. Retrying in {wait_time}s... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait_time)
                return await fetch_user_details(session, agent_id, retry_count + 1)
            else:
                error_text = await resp.text()
                if LOG_502_DETAILS or resp.status != 502:
                    logger.error(f"Fetch Details Error {resp.status} for Agent {agent_id}: {error_text[:200]}")
                else:
                    logger.error(f"Fetch Details Error {resp.status} for Agent {agent_id} - Server unavailable")
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout fetching user details for Agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ Fetch Details Exception for Agent {agent_id}: {e}")
    return []

async def process_single_user(session, user, session_dir, agent_id):
    """Step 3 & 4: Process via Local AI and Push Result"""
    user_id = user.get("user_id")
    
    user_log = {
        "user_id": user_id,
        "agent_id": agent_id,
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
                    logger.error(f" User {user_id}: Local AI Failed ({resp.status})")
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
                    "ai_score": ai_result.get("score", 0),
                    "ai_breakdown": ai_result.get("breakdown", {}),
                    "ai_rejection_reasons": ai_result.get("rejection_reasons", []),
                    "ai_extracted_data": ai_result.get("extracted_data", {}),
                    "ai_result": ai_result
                })
                
                # Log detailed decision information
                decision_details = f"Decision: {ai_result.get('final_decision')} | Score: {ai_result.get('score', 0)}"
                if ai_result.get('rejection_reasons'):
                    decision_details += f" | Reasons: {', '.join(ai_result.get('rejection_reasons', []))}"
                logger.info(f"  User {user_id}: {decision_details}")
                
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
            logger.error(f" User {user_id}: Local AI Timeout")
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
            "agent_id": str(agent_id),
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
                        "server_response": push_result,
                        "pushed_data": {
                            "final_decision": push_payload['final_decision'],
                            "status_code": push_payload['status_code'],
                            "rejection_reasons": push_payload['rejection_reasons'],
                            "extracted_data": push_payload['extracted_data']
                        }
                    })
                    
                    # Log with detailed information
                    decision_log = f"{push_payload['final_decision']} (Code: {push_payload['status_code']})"
                    if push_payload['rejection_reasons']:
                        decision_log += f" | Reasons: {', '.join(push_payload['rejection_reasons'])}"
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
                    
                    logger.error(f" User {user_id}: Failed to Push Result {push_resp.status} - {error_text[:100]}")
                    
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
            logger.error(f" User {user_id}: Push Exception - {push_error}")
        
        # Store final result with detailed breakdown
        user_log["final_result"] = {
            "final_decision": ai_result.get("final_decision", "REJECTED"),
            "status_code": ai_result.get("status_code", 0),
            "score": ai_result.get("score", 0),
            "breakdown": ai_result.get("breakdown", {}),
            "extracted_data": ai_result.get("extracted_data", {}),
            "rejection_reasons": ai_result.get("rejection_reasons", []),
            "decision_summary": {
                "is_approved": ai_result.get("final_decision") == "APPROVED",
                "is_rejected": ai_result.get("final_decision") == "REJECTED",
                "is_review": ai_result.get("final_decision") == "REVIEW",
                "total_issues": len(ai_result.get("rejection_reasons", [])),
                "extracted_aadhaar": ai_result.get("extracted_data", {}).get("aadhaar"),
                "extracted_dob": ai_result.get("extracted_data", {}).get("dob"),
                "extracted_gender": ai_result.get("extracted_data", {}).get("gender")
            }
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
        logger.error(f" User {user_id} Exception: {e}")
    
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

async def process_agent_batch(session, agent_id, session_dir):
    """Process a single batch for a specific agent"""
    logger.info(f"🔄 --- Starting Cycle for Agent {agent_id} ---")
    
    # 1. Lock Batch
    locked_ids = await lock_batch(session, agent_id)
    
    if not locked_ids:
        logger.warning(f"⚠️ No users available to lock for Agent {agent_id} (Server may be down or no pending users)")
        return False
    
    # 2. Fetch User Details
    users_to_process = await fetch_user_details(session, agent_id)
    
    if not users_to_process:
        logger.warning(f"Locked batch for Agent {agent_id} but got no user details. Releasing batch...")
        await release_batch(session, agent_id)
        return False
    
    # 3. Process Batch
    batch_start_time = time.time()
    batch_results = []
    
    # Process in chunks of 5 users at a time
    chunk_size = 5
    for i in range(0, len(users_to_process), chunk_size):
        chunk = users_to_process[i:i + chunk_size]
        tasks = [process_single_user(session, u, session_dir, agent_id) for u in chunk]
        chunk_results = await asyncio.gather(*tasks)
        batch_results.extend(chunk_results)
    
    batch_elapsed = time.time() - batch_start_time
    
    # Create batch summary with detailed information
    batch_summary = []
    for user_log in batch_results:
        if user_log:
            final_result = user_log["final_result"]
            batch_summary.append({
                "user_id": user_log["user_id"],
                "final_decision": final_result.get("final_decision"),
                "status_code": final_result.get("status_code"),
                "score": final_result.get("score", 0),
                "breakdown": final_result.get("breakdown", {}),
                "rejection_reasons": final_result.get("rejection_reasons", []),
                "extracted_data": final_result.get("extracted_data", {}),
                "processing_time": user_log["timing"].get("total_time"),
                "errors": len(user_log["errors"]),
                "has_errors": len(user_log["errors"]) > 0
            })
    
    # Save batch summary
    save_batch_summary(session_dir, {
        "batch_elapsed_time": batch_elapsed,
        "users": batch_summary
    })
    
    # Calculate and log batch statistics
    approved = sum(1 for u in batch_summary if u['final_decision'] == 'APPROVED')
    rejected = sum(1 for u in batch_summary if u['final_decision'] == 'REJECTED')
    review = sum(1 for u in batch_summary if u['final_decision'] == 'REVIEW')
    
    logger.info(f"--- Batch Complete for Agent {agent_id}. Processed {len(batch_results)} users in {batch_elapsed:.2f}s ---")
    logger.info(f"📊 Results: ✅ {approved} Approved | ❌ {rejected} Rejected | ⏳ {review} Review")
    
    # Log common rejection reasons if any
    all_rejection_reasons = []
    for u in batch_summary:
        all_rejection_reasons.extend(u.get('rejection_reasons', []))
    
    if all_rejection_reasons:
        from collections import Counter
        reason_counts = Counter(all_rejection_reasons)
        logger.info(f"🔍 Top Rejection Reasons:")
        for reason, count in reason_counts.most_common(5):
            logger.info(f"   • {reason}: {count} times")
    
    # 4. Release the batch for this agent
    await release_batch(session, agent_id)
    
    return True

async def run_pipeline():
    """Main pipeline that cycles through all agents continuously"""
    async with aiohttp.ClientSession() as session:
        try:
            agent_sessions = {}
            
            # Create session directories for each agent
            for agent_id in AGENT_IDS:
                session_dir = create_session_log_file(agent_id)
                agent_sessions[agent_id] = session_dir
                logger.info(f"📁 Agent {agent_id} logs will be saved to: {session_dir}")
            
            logger.info(f"🚀 Starting continuous processing loop for Agents: {AGENT_IDS}")
            
            while True:
                # Cycle through each agent
                for agent_id in AGENT_IDS:
                    session_dir = agent_sessions[agent_id]
                    
                    # Process one batch for this agent
                    processed = await process_agent_batch(session, agent_id, session_dir)
                    
                    if not processed:
                        logger.info(f"⏭️ No work available for Agent {agent_id}, moving to next agent...")
                    
                    # Delay between agents to avoid rate limiting
                    await asyncio.sleep(AGENT_CYCLE_DELAY)
                
                # After processing all agents, take a longer break before the next cycle
                logger.info(f"--- Completed cycle for all agents. Starting next cycle in {FULL_CYCLE_DELAY}s... ---")
                await asyncio.sleep(FULL_CYCLE_DELAY)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested...")
            raise
        finally:
            # Save session summaries for all agents before exiting
            logger.info("💾 Saving final session summaries for all agents...")
            for agent_id, session_dir in agent_sessions.items():
                save_session_summary(session_dir)
                logger.info(f"✅ Agent {agent_id} logs saved to: {session_dir}")

if __name__ == "__main__":
    print(f"🚀 Starting Production Dispatcher for Agents: {AGENT_IDS}")
    print(f"📋 Will process agents in rotation continuously")
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")