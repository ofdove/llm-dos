#!/usr/bin/env python3
"""
Script to send all normal prompts from chatdoctor/prompt/ to vllm server.
"""
import os
import json
import requests
import glob
from tqdm import tqdm
import subprocess
import signal
import threading
import csv
import time
from datetime import datetime

# Configuration
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT_DIR = "../chatdoctor/prompt"
MAX_PROMPTS = 1
CUDA_TRACE_PATH = "../libbpf-bootstrap/examples/tracing/cuda_trace"
OUTPUT_DIR = "../tracing_output"

def get_vllm_pid():
    """Get the PID of the vLLM process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "EngineCor"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            # Return the first PID (main vLLM process)
            return int(pids[0])
        return None
    except Exception as e:
        print(f"Error getting vLLM PID: {e}")
        return None

class CudaTracer:
    """Manages CUDA tracing process."""
    def __init__(self, pid, output_file):
        self.pid = pid
        self.output_file = output_file
        self.process = None
        self.thread = None
        self.output_path = None
        self.running = False
        self.ready_event = threading.Event()  # Signal when tracing is ready
        self.error_occurred = False
        
    def start(self):
        """Start CUDA tracing and wait for it to be ready."""
        if not self.pid:
            print("Warning: No vLLM PID found, skipping CUDA tracing")
            return False
        
        cuda_trace_full_path = os.path.abspath(CUDA_TRACE_PATH)
        if not os.path.exists(cuda_trace_full_path):
            print(f"Warning: cuda_trace not found at {cuda_trace_full_path}, skipping CUDA tracing")
            return False
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.output_path = os.path.join(OUTPUT_DIR, self.output_file)
        
        print(f"\n{'='*60}")
        print(f"Starting CUDA tracing for PID {self.pid}")
        print(f"Output will be saved to: {self.output_path}")
        print(f"{'='*60}")
        print("Please enter your sudo password when prompted...")
        print(f"{'='*60}\n")
        
        def trace_thread():
            """Run cuda_trace in a separate thread."""
            try:
                # Run cuda_trace with sudo (stdin=None allows password prompt)
                cmd = ["sudo", cuda_trace_full_path, "--pid", str(self.pid)]
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=None,  # Allow stdin for password input
                    text=True,
                    bufsize=1
                )
                self.running = True
                
                # Read output line by line and write to CSV
                with open(self.output_path, 'w', newline='') as csvfile:
                    writer = None
                    header_written = False
                    
                    for line in self.process.stdout:
                        if not self.running:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Skip the "running.. press Ctrl-C to stop" line
                        if "running" in line.lower() or "press" in line.lower():
                            continue
                        
                        # First data line should be the header
                        if not header_written:
                            headers = line.split(',')
                            writer = csv.writer(csvfile)
                            writer.writerow(headers)
                            header_written = True
                            # Signal that tracing has started successfully
                            self.ready_event.set()
                            print("✓ CUDA tracing started successfully!")
                        else:
                            # Write data rows
                            if writer:
                                writer.writerow(line.split(','))
                
                if self.process:
                    self.process.wait()
                print(f"CUDA tracing completed. Output saved to {self.output_path}")
                
            except Exception as e:
                print(f"Error running cuda_trace: {e}")
                self.error_occurred = True
                self.ready_event.set()  # Set event even on error to unblock main thread
            finally:
                self.running = False
        
        self.thread = threading.Thread(target=trace_thread, daemon=True)
        self.thread.start()
        
        # Wait for tracing to start (with timeout)
        print("Waiting for CUDA tracing to initialize...")
        if self.ready_event.wait(timeout=30):  # Wait up to 30 seconds
            if self.error_occurred:
                print("Failed to start CUDA tracing. Continuing without tracing...")
                return False
            return True
        else:
            print("Timeout waiting for CUDA tracing to start. Continuing without tracing...")
            return False
    
    def stop(self):
        """Stop CUDA tracing process."""
        if self.process and self.running:
            self.running = False
            try:
                # Send SIGINT to gracefully stop cuda_trace
                self.process.send_signal(signal.SIGINT)
                # Wait a bit for it to finish
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop gracefully
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"Error stopping cuda_trace: {e}")
            print(f"CUDA tracing stopped. Output saved to {self.output_path}")

def send_prompt_to_vllm(prompt, request_id=None):
    """Send a prompt to the vllm server."""
    headers = {
        'Content-Type': 'application/json',
    }
    if request_id:
        headers['X-Request-ID'] = request_id
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(VLLM_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending prompt: {e}")
        return None

def main():
    # Get vLLM PID for CUDA tracing
    vllm_pid = get_vllm_pid()
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"normal_{timestamp}.csv"
    
    # Start CUDA tracing (will wait for sudo password and confirmation)
    tracer = None
    if vllm_pid:
        tracer = CudaTracer(vllm_pid, output_csv)
        tracer.start()  # This will wait for password and confirmation
    
    # Get all JSON files from prompt directory
    prompt_files = sorted(glob.glob(os.path.join(PROMPT_DIR, "*.json")))
    
    if not prompt_files:
        print(f"No JSON files found in {PROMPT_DIR}")
        return
    
    # Limit to first MAX_PROMPTS files
    prompt_files = prompt_files[:MAX_PROMPTS]
    
    print(f"Found {len(prompt_files)} prompt files (limited to {MAX_PROMPTS})")
    print(f"Sending prompts to {VLLM_URL}")
    if vllm_pid:
        print(f"CUDA tracing active for PID {vllm_pid}")
    print("-" * 50)
    
    # Process each prompt file
    for prompt_file in tqdm(prompt_files, desc="Sending prompts"):
        try:
            # Read the prompt from JSON file
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
            
            # Extract the prompt (first element of the array)
            if isinstance(prompt_data, list) and len(prompt_data) > 0:
                prompt = prompt_data[0]
            else:
                print(f"Warning: {prompt_file} has unexpected format, skipping")
                continue
            
            # Generate request ID from filename
            filename = os.path.basename(prompt_file)
            request_id = f"normal-{filename.replace('.json', '')}"
            
            # Send to vllm server
            result = send_prompt_to_vllm(prompt, request_id=request_id)
            
            if result:
                # Optionally print the response (commented out to avoid clutter)
                # print(f"\nResponse for {filename}:")
                # print(json.dumps(result, indent=2))
                pass
            else:
                print(f"Failed to send {filename}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {prompt_file}: {e}")
        except Exception as e:
            print(f"Error processing {prompt_file}: {e}")
    
    # Stop CUDA tracing
    if tracer:
        print("\nWaiting for CUDA operations to complete...")
        # Wait a bit longer to allow async CUDA operations to complete
        # CUDA operations are asynchronous and may still be running after HTTP response
        time.sleep(10)  # Wait 5 seconds for pending operations
        print("Stopping CUDA tracing...")
        tracer.stop()
    
    print("\nDone!")

if __name__ == "__main__":
    main()

