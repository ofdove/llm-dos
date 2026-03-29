#!/usr/bin/env python3
"""
Simple script to send prompts to vLLM server without CUDA tracing integration.
"""
import os
import json
import requests
import glob
from tqdm import tqdm

# Configuration
VLLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT_DIR = "../chatdoctor/prompt"
MAX_PROMPTS = 1

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
    # Get all JSON files from prompt directory
    prompt_files = sorted(glob.glob(os.path.join(PROMPT_DIR, "*.json")))
    
    if not prompt_files:
        print(f"No JSON files found in {PROMPT_DIR}")
        return
    
    # Limit to first MAX_PROMPTS files
    prompt_files = prompt_files[:MAX_PROMPTS]
    
    print(f"Found {len(prompt_files)} prompt files (limited to {MAX_PROMPTS})")
    print(f"Sending prompts to {VLLM_URL}")
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
            request_id = f"prompt-{filename.replace('.json', '')}"
            
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
    
    print("\nDone!")

if __name__ == "__main__":
    main()

