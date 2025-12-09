import subprocess
import sys
import time
import os

# Configuration
START_PORT = 8100
NUM_AGENTS = 4  # Number of parallel agents (Be careful of GPU VRAM!)

processes = []

def start_agents():
    print(f"🚀 Launching {NUM_AGENTS} Verification Agents...")
    
    for i in range(NUM_AGENTS):
        port = START_PORT + i
        agent_id = str(i + 1)
        
        # Command to run uvicorn
        # We pass the AGENT_ID as an env var so main.py can read it if needed
        env = os.environ.copy()
        env["AGENT_ID"] = agent_id
        
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(port),
            "--workers", "1" # Keep 1 worker per port (since we have multiple ports)
        ]
        
        print(f"   [Agent {agent_id}] Starting on Port {port}...")
        
        # Start process in background
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
        # Stagger start times slightly to prevent GPU memory spike collision
        time.sleep(2) 

    print("\n✅ All Agents Running. Press Ctrl+C to stop.")
    
    try:
        # Keep script running to monitor child processes
        while True:
            time.sleep(1)
            # Check if any process died
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"⚠️  Agent {i+1} died! Restarting...")
                    # logic to restart could go here
                    break
    except KeyboardInterrupt:
        stop_agents()

def stop_agents():
    print("\n🛑 Stopping all agents...")
    for p in processes:
        p.terminate()
    print("Cleaned up.")

if __name__ == "__main__":
    start_agents()