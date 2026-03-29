#!/bin/bash
#
# Script to send ALL prompts (DOS or normal) to vLLM server with CUDA tracing for each
# Each prompt gets its own CUDA trace file named after the JSON file
# Usage: ./send_all_prompts.sh [dos|normal] [start_number] [max_prompts]
# Example: ./send_all_prompts.sh normal 723 100  # Start from 723.json, process 100 prompts
#

set -e  # Exit on error

# Configuration
VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
PROMPT_TYPE="${1:-normal}"  # Default to "normal" if no argument provided
START_FROM="${2:-1}"        # Start from this prompt number (default: 1)
MAX_PROMPTS="${3:-999999}"  # Maximum number of prompts to process (default: all)
CUDA_TRACE_PATH="../libbpf-bootstrap/examples/tracing/cuda_trace"
BASE_OUTPUT_DIR="../tracing_output"

# Set prompt directory and output directory based on type
if [ "$PROMPT_TYPE" = "dos" ]; then
    PROMPT_DIR="../chatdoctor/dos-prompt"
    PREFIX="dos"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/dos"
elif [ "$PROMPT_TYPE" = "normal" ]; then
    PROMPT_DIR="../chatdoctor/prompt"
    PREFIX="normal"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/normal"
else
    echo "Error: Invalid prompt type '$PROMPT_TYPE'"
    echo "Usage: $0 [dos|normal]"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to get vLLM PID
get_vllm_pid() {
    local pid=$(pgrep -f "EngineCor" | head -n 1)
    if [ -z "$pid" ]; then
        echo ""
    else
        echo "$pid"
    fi
}

# Function to start CUDA tracing
start_cuda_tracing() {
    local pid=$1
    local output_file=$2

    if [ -z "$pid" ]; then
        echo -e "${YELLOW}Warning: No vLLM PID found, skipping CUDA tracing${NC}" >&2
        return 1
    fi

    local cuda_trace_full_path=$(realpath "$CUDA_TRACE_PATH" 2>/dev/null || echo "$CUDA_TRACE_PATH")

    if [ ! -f "$cuda_trace_full_path" ]; then
        echo -e "${YELLOW}Warning: cuda_trace not found at $cuda_trace_full_path, skipping CUDA tracing${NC}" >&2
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"
    local output_path="$OUTPUT_DIR/$output_file"

    # Start CUDA tracing in background with output redirection
    sudo "$cuda_trace_full_path" --pid "$pid" > "$output_path" 2>&1 &
    local trace_pid=$!

    # Wait a moment for tracing to start
    sleep 0.5

    # Check if process is still running
    if kill -0 "$trace_pid" 2>/dev/null; then
        # Return both PID and output path (separated by |) to stdout only
        echo "$trace_pid|$output_path"
        return 0
    else
        echo -e "${RED}Failed to start CUDA tracing${NC}" >&2
        return 1
    fi
}

# Function to stop CUDA tracing
stop_cuda_tracing() {
    local trace_info=$1

    if [ -z "$trace_info" ]; then
        return
    fi

    # Parse PID and output path (format: "PID|output_path")
    local trace_pid=$(echo "$trace_info" | cut -d'|' -f1)
    local output_path=$(echo "$trace_info" | cut -d'|' -f2)

    if kill -0 "$trace_pid" 2>/dev/null; then
        # Wait for async CUDA operations to complete
        sleep 10

        # Send SIGINT to gracefully stop
        sudo kill -INT "$trace_pid" 2>/dev/null || true

        # Wait for process to finish (with timeout)
        local wait_count=0
        while kill -0 "$trace_pid" 2>/dev/null && [ $wait_count -lt 30 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # Force kill if still running
        if kill -0 "$trace_pid" 2>/dev/null; then
            echo -e "${YELLOW}Warning: Force killing cuda_trace process${NC}" >&2
            sudo kill -9 "$trace_pid" 2>/dev/null || true
            sleep 1
        fi

        # Remove first line from CSV file if it exists and has content
        if [ -f "$output_path" ] && [ -s "$output_path" ]; then
            # Use tail to skip first line (more reliable than sed -i)
            tail -n +2 "$output_path" > "${output_path}.tmp" && mv "${output_path}.tmp" "$output_path"
        fi
    else
        # Even if process is not running, try to remove first line if file exists
        if [ -n "$output_path" ] && [ -f "$output_path" ] && [ -s "$output_path" ]; then
            tail -n +2 "$output_path" > "${output_path}.tmp" && mv "${output_path}.tmp" "$output_path"
        fi
    fi
}

# Function to send prompt to vLLM
send_prompt_to_vllm() {
    local prompt="$1"
    local request_id="$2"

    local headers="Content-Type: application/json"
    if [ -n "$request_id" ]; then
        headers="$headers\nX-Request-ID: $request_id"
    fi

    local payload=$(cat <<EOF
{
  "messages": [
    {"role": "user", "content": $(echo "$prompt" | jq -Rs .)}
  ]
}
EOF
)

    local response=$(curl -s -w "\n%{http_code}" \
        -X POST "$VLLM_URL" \
        -H "Content-Type: application/json" \
        ${request_id:+-H "X-Request-ID: $request_id"} \
        -d "$payload")

    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    if [ "$http_code" -eq 200 ]; then
        echo "$body"
        return 0
    else
        echo -e "${RED}Error: HTTP $http_code${NC}" >&2
        echo "$body" >&2
        return 1
    fi
}

# Main execution
main() {
    echo "============================================================"
    echo "Sending ALL $PROMPT_TYPE prompts to vLLM"
    echo "Each prompt will get its own CUDA trace file"
    echo "============================================================"

    # Get vLLM PID
    vllm_pid=$(get_vllm_pid)

    if [ -n "$vllm_pid" ]; then
        echo "Found vLLM PID: $vllm_pid"
        echo ""
        echo "============================================================"
        echo "Please enter your sudo password for CUDA tracing..."
        echo "============================================================"

        # Pre-authenticate sudo once at the beginning
        if ! sudo -v; then
            echo -e "${RED}Failed to authenticate sudo${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Sudo authenticated${NC}"
    else
        echo -e "${YELLOW}Warning: No vLLM PID found, will skip CUDA tracing${NC}"
    fi

    # Get all JSON files from prompt directory
    if [ ! -d "$PROMPT_DIR" ]; then
        echo -e "${RED}Error: Prompt directory not found: $PROMPT_DIR${NC}"
        exit 1
    fi

    prompt_files=($(find "$PROMPT_DIR" -name "*.json" -type f | sort -V))

    if [ ${#prompt_files[@]} -eq 0 ]; then
        echo -e "${RED}No JSON files found in $PROMPT_DIR${NC}"
        exit 1
    fi

    echo ""
    echo "Found ${#prompt_files[@]} prompt file(s)"
    echo "Sending prompts to $VLLM_URL"
    echo "Output directory: $OUTPUT_DIR"
    echo "------------------------------------------------------------"
    echo ""

    # Process each prompt file
    success_count=0
    fail_count=0
    skip_count=0

    for prompt_file in "${prompt_files[@]}"; do
        filename=$(basename "$prompt_file")
        base_filename="${filename%.json}"

        # Extract numeric part from filename for comparison
        file_number=$(echo "$base_filename" | sed 's/[^0-9]*//g')

        # Skip if before START_FROM
        if [ -n "$file_number" ] && [ "$file_number" -lt "$START_FROM" ]; then
            skip_count=$((skip_count + 1))
            continue
        fi

        # Skip if CSV already exists
        output_csv="${base_filename}.csv"
        if [ -f "$OUTPUT_DIR/$output_csv" ]; then
            echo "Skipping $filename (CSV already exists)"
            skip_count=$((skip_count + 1))
            continue
        fi

        echo "============================================================"
        echo "Processing: $filename"
        echo "============================================================"

        # Read and parse JSON file
        if ! command -v jq &> /dev/null; then
            echo -e "${RED}Error: jq is required but not installed${NC}"
            echo "Install with: sudo apt-get install jq"
            exit 1
        fi

        # Extract prompt (first element of array)
        prompt=$(jq -r '.[0] // .' "$prompt_file" 2>/dev/null)

        if [ -z "$prompt" ] || [ "$prompt" = "null" ]; then
            echo -e "${YELLOW}Warning: Unexpected format, skipping${NC}"
            fail_count=$((fail_count + 1))
            continue
        fi

        # Start CUDA tracing for this specific prompt with filename-based CSV
        trace_pid=""
        if [ -n "$vllm_pid" ]; then
            echo "Starting CUDA trace: $output_csv"
            trace_pid=$(start_cuda_tracing "$vllm_pid" "$output_csv")
            if [ -n "$trace_pid" ]; then
                echo -e "${GREEN}✓ CUDA tracing started${NC}"
            fi
        fi

        # Generate request ID from filename
        request_id="${PREFIX}-${base_filename}"

        # Send to vLLM server
        echo "Sending prompt to vLLM..."
        if send_prompt_to_vllm "$prompt" "$request_id" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Prompt sent successfully${NC}"
            success_count=$((success_count + 1))
        else
            echo -e "${RED}✗ Failed to send prompt${NC}"
            fail_count=$((fail_count + 1))
        fi

        # Stop CUDA tracing for this prompt
        if [ -n "$trace_pid" ]; then
            echo "Stopping CUDA trace..."
            stop_cuda_tracing "$trace_pid"
            echo -e "${GREEN}✓ CUDA trace saved: $OUTPUT_DIR/$output_csv${NC}"
        fi

        echo ""
    done

    echo "============================================================"
    echo "Done!"
    echo "Success: $success_count, Failed: $fail_count, Skipped: $skip_count"
    echo "Output directory: $OUTPUT_DIR"
    echo "============================================================"
}

# Run main function
main
