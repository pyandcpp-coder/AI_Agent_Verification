import asyncio
import aiohttp
import time
import random

# Configuration
AGENT_PORTS = [8100, 8101, 8102, 8103]
API_ENDPOINT = "/verification/verify/agent/"

# Mock Data Generator (Replace this with your DB fetch logic)
def fetch_next_batch(batch_size=300):
    print(f"\n📥 Fetching next batch of {batch_size} users...")
    batch = []
    for i in range(batch_size):
        # Create dummy request matching your VerifyRequest model
        user = {
            "user_id": 38000 + i,
            "dob": "06-12-2007",
            "passport_first": "uploads/sample_front.jpg", # Ensure these exist or use URLs
            "passport_old": "uploads/sample_back.jpg",
            "selfie_photo": "uploads/sample_selfie.jpg",
            "gender": "Male"
        }
        batch.append(user)
    return batch

async def send_request(session, url, data):
    try:
        async with session.post(url, json=data, timeout=60) as response:
            result = await response.json()
            return result
    except Exception as e:
        return {"user_id": data["user_id"], "status": "ERROR", "message": str(e)}

async def process_batch(batch_data):
    results = []
    
    # Connection pool
    connector = aiohttp.TCPConnector(limit=100) # High limit, we handle concurrency manually
    async with aiohttp.ClientSession(connector=connector) as session:
        
        tasks = []
        for i, user_data in enumerate(batch_data):
            # Round Robin Distribution
            # User 1 -> Port 8100, User 2 -> Port 8101, etc.
            assigned_port = AGENT_PORTS[i % len(AGENT_PORTS)]
            url = f"http://localhost:{assigned_port}{API_ENDPOINT}"
            
            task = asyncio.create_task(send_request(session, url, user_data))
            tasks.append(task)
        
        print(f"🔥 Dispatching {len(tasks)} requests across {len(AGENT_PORTS)} agents...")
        start_time = time.time()
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        print(f"✅ Batch Completed in {duration:.2f}s")
        print(f"⚡ Throughput: {len(batch_data)/duration:.2f} req/s")
        
        return responses

async def main_loop():
    while True:
        # 1. Get 300 Users
        batch = fetch_next_batch(300)
        
        if not batch:
            print("No more users to process. Exiting.")
            break

        # 2. Process them
        results = await process_batch(batch)
        
        # 3. Handle Results (Save to DB, print, etc.)
        approved = sum(1 for r in results if r.get('final_decision') == 'APPROVED')
        rejected = sum(1 for r in results if r.get('final_decision') == 'REJECTED')
        errors = sum(1 for r in results if r.get('status') == 'ERROR')
        
        print(f"📊 Report: Approved: {approved} | Rejected: {rejected} | Errors: {errors}")
        
        # 4. Wait/Sleep before next batch if needed
        # time.sleep(5) 
        
        # For demo purposes, break after 1 batch
        break 

if __name__ == "__main__":
    asyncio.run(main_loop())