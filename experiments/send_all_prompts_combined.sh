#!/bin/bash
#
# Combined script to send ALL prompts (DOS or normal) to vLLM server
# with BOTH CPU and CUDA tracing running simultaneously for each prompt.
# Each prompt gets its own CPU trace file and CUDA trace file named after the JSON file.
# Usage: ./send_all_prompts_combined.sh [dos|normal] [start_number] [max_prompts]
# Example: ./send_all_prompts_combined.sh normal 723 100  # Start from 723.json, process 100 prompts
#

set -e  # Exit on error

# Configuration
VLLM_URL="http://127.0.0.1:8000/v1/chat/completions"
PROMPT_TYPE="${1:-normal}"  # Default to "normal" if no argument provided
START_FROM="${2:-1}"        # Start from this prompt number (default: 1)
MAX_PROMPTS="${3:-999999}"  # Maximum number of prompts to process (default: all)

# Tracer binary paths
CUDA_TRACE_PATH="../libbpf-bootstrap/examples/tracing/cuda_trace"
CPU_TRACE_PATH="../libbpf-bootstrap/examples/tracing/cpu_trace"

# Output base directories
CUDA_BASE_OUTPUT_DIR="../tracing_output"
CPU_BASE_OUTPUT_DIR="../tracing_output_cpu"

# Set prompt directory and output directories based on type
if [ "$PROMPT_TYPE" = "dos" ]; then
    PROMPT_DIR="../chatdoctor/dos-prompt"
    PREFIX="dos"
    CUDA_OUTPUT_DIR="$CUDA_BASE_OUTPUT_DIR/dos"
    CPU_OUTPUT_DIR="$CPU_BASE_OUTPUT_DIR/dos"
elif [ "$PROMPT_TYPE" = "normal" ]; then
    PROMPT_DIR="../chatdoctor/prompt"
    PREFIX="normal"
    CUDA_OUTPUT_DIR="$CUDA_BASE_OUTPUT_DIR/normal"
    CPU_OUTPUT_DIR="$CPU_BASE_OUTPUT_DIR/normal"
else
    echo "Error: Invalid prompt type '$PROMPT_TYPE'"
    echo "Usage: $0 [dos|normal] [start_number] [max_prompts]"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
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

# Generic function to start a tracer (works for both CPU and CUDA)
# Arguments: $1=vllm_pid, $2=output_file, $3=tracer_binary_path, $4=output_directory, $5=tracer_label
start_tracing() {
    local pid=$1
    local output_file=$2
    local tracer_path=$3
    local output_dir=$4
    local label=$5

    if [ -z "$pid" ]; then
        echo -e "${YELLOW}Warning: No vLLM PID found, skipping $label tracing${NC}" >&2
        return 1
    fi

    local tracer_full_path=$(realpath "$tracer_path" 2>/dev/null || echo "$tracer_path")

    if [ ! -f "$tracer_full_path" ]; then
        echo -e "${YELLOW}Warning: $label tracer not found at $tracer_full_path, skipping${NC}" >&2
        return 1
    fi

    # Ensure output directory exists
    mkdir -p "$output_dir"
    local output_path="$output_dir/$output_file"

    # Start tracing in background with output redirection
    sudo "$tracer_full_path" --pid "$pid" > "$output_path" 2>&1 &
    local trace_pid=$!

    # Wait a moment for tracing to start
    sleep 0.5

    # Check if process is still running
    if kill -0 "$trace_pid" 2>/dev/null; then
        # Return both PID and output path (separated by |) to stdout only
        echo "$trace_pid|$output_path"
        return 0
    else
        echo -e "${RED}Failed to start $label tracing${NC}" >&2
        return 1
    fi
}

# Wait until GPU utilization drops to near-idle (or timeout)
wait_for_gpu_idle() {
    local timeout_secs="${1:-60}"
    local threshold="${2:-5}"  # % GPU utilization considered "idle"
    local elapsed=0

    if ! command -v nvidia-smi &>/dev/null; then
        sleep 30
        return
    fi

    echo "Waiting for GPU to go idle (util < ${threshold}%, timeout ${timeout_secs}s)..." >&2
    while [ "$elapsed" -lt "$timeout_secs" ]; do
        local util
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
        if [ -n "$util" ] && [ "$util" -le "$threshold" ] 2>/dev/null; then
            echo "GPU idle (util=${util}%) after ${elapsed}s" >&2
            return
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "GPU idle wait timed out after ${timeout_secs}s, proceeding anyway" >&2
}

# Generic function to stop a tracer
# Arguments: $1=trace_info (PID|path), $2=tracer_label
stop_tracing() {
    local trace_info=$1
    local label=$2

    if [ -z "$trace_info" ]; then
        return
    fi

    # Parse PID and output path (format: "PID|output_path")
    local trace_pid=$(echo "$trace_info" | cut -d'|' -f1)
    local output_path=$(echo "$trace_info" | cut -d'|' -f2)

    if kill -0 "$trace_pid" 2>/dev/null; then
        # Wait for GPU to finish all async CUDA operations before stopping tracer
        wait_for_gpu_idle 120 5

        # Send SIGINT to gracefully stop
        sudo kill -INT "$trace_pid" 2>/dev/null || true

        # Wait for process to finish (with timeout)
        local wait_count=0
        while kill -0 "$trace_pid" 2>/dev/null && [ $wait_count -lt 60 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done

        # Force kill if still running
        if kill -0 "$trace_pid" 2>/dev/null; then
            echo -e "${YELLOW}Warning: Force killing $label trace process${NC}" >&2
            sudo kill -9 "$trace_pid" 2>/dev/null || true
            sleep 1
        fi

        # Remove first line from CSV file if it exists and has content
        if [ -f "$output_path" ] && [ -s "$output_path" ]; then
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
    echo "Each prompt will get BOTH CPU and CUDA trace files"
    echo "============================================================"

    # Get vLLM PID
    vllm_pid=$(get_vllm_pid)

    if [ -n "$vllm_pid" ]; then
        echo "Found vLLM PID: $vllm_pid"
        echo ""

        # Resolve and display tracer paths
        cuda_trace_resolved=$(realpath "$CUDA_TRACE_PATH" 2>/dev/null || echo "$CUDA_TRACE_PATH")
        cpu_trace_resolved=$(realpath "$CPU_TRACE_PATH" 2>/dev/null || echo "$CPU_TRACE_PATH")
        echo "CUDA tracer: $cuda_trace_resolved"
        echo "CPU  tracer: $cpu_trace_resolved"
        echo ""

        echo "============================================================"
        echo "Please enter your sudo password for tracing..."
        echo "============================================================"

        # Pre-authenticate sudo once at the beginning
        if ! sudo -v; then
            echo -e "${RED}Failed to authenticate sudo${NC}"
            exit 1
        fi
        echo -e "${GREEN}Sudo authenticated${NC}"
    else
        echo -e "${YELLOW}Warning: No vLLM PID found, will skip tracing${NC}"
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
    echo "CUDA output directory: $CUDA_OUTPUT_DIR"
    echo "CPU  output directory: $CPU_OUTPUT_DIR"
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

        # Check max prompts limit (only count non-skipped prompts)
        processed=$((success_count + fail_count))
        if [ "$processed" -ge "$MAX_PROMPTS" ]; then
            echo "Reached max prompts limit ($MAX_PROMPTS), stopping."
            break
        fi

        # Skip if BOTH CSV files already exist
        output_csv="${base_filename}.csv"
        cuda_csv_exists=false
        cpu_csv_exists=false
        if [ -f "$CUDA_OUTPUT_DIR/$output_csv" ]; then
            cuda_csv_exists=true
        fi
        if [ -f "$CPU_OUTPUT_DIR/$output_csv" ]; then
            cpu_csv_exists=true
        fi
        if [ "$cuda_csv_exists" = true ] && [ "$cpu_csv_exists" = true ]; then
            echo "Skipping $filename (both CPU and CUDA CSVs already exist)"
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

        # Start BOTH tracers simultaneously
        cuda_trace_info=""
        cpu_trace_info=""

        if [ -n "$vllm_pid" ]; then
            # Start CUDA tracing (skip if CSV already exists)
            if [ "$cuda_csv_exists" = false ]; then
                echo "Starting CUDA trace: $output_csv"
                cuda_trace_info=$(start_tracing "$vllm_pid" "$output_csv" "$CUDA_TRACE_PATH" "$CUDA_OUTPUT_DIR" "CUDA")
                if [ -n "$cuda_trace_info" ]; then
                    echo -e "${GREEN}CUDA tracing started${NC}"
                fi
            else
                echo -e "${CYAN}CUDA CSV already exists, skipping CUDA tracer${NC}"
            fi

            # Start CPU tracing (skip if CSV already exists)
            if [ "$cpu_csv_exists" = false ]; then
                echo "Starting CPU  trace: $output_csv"
                cpu_trace_info=$(start_tracing "$vllm_pid" "$output_csv" "$CPU_TRACE_PATH" "$CPU_OUTPUT_DIR" "CPU")
                if [ -n "$cpu_trace_info" ]; then
                    echo -e "${GREEN}CPU  tracing started${NC}"
                fi
            else
                echo -e "${CYAN}CPU CSV already exists, skipping CPU tracer${NC}"
            fi
        fi

        # Generate request ID from filename
        request_id="${PREFIX}-${base_filename}"

        # Send prompt to vLLM (only once for both tracers)
        echo "Sending prompt to vLLM..."
        if send_prompt_to_vllm "$prompt" "$request_id" > /dev/null 2>&1; then
            echo -e "${GREEN}Prompt sent successfully${NC}"
            success_count=$((success_count + 1))
        else
            echo -e "${RED}Failed to send prompt${NC}"
            fail_count=$((fail_count + 1))
        fi

        # Stop BOTH tracers
        # Note: We stop them sequentially. Each stop_tracing call waits ~10s for async
        # operations, so both tracers capture the full inference window.
        if [ -n "$cuda_trace_info" ]; then
            echo "Stopping CUDA trace..."
            stop_tracing "$cuda_trace_info" "CUDA"
            echo -e "${GREEN}CUDA trace saved: $CUDA_OUTPUT_DIR/$output_csv${NC}"
        fi

        if [ -n "$cpu_trace_info" ]; then
            echo "Stopping CPU  trace..."
            stop_tracing "$cpu_trace_info" "CPU"
            echo -e "${GREEN}CPU  trace saved: $CPU_OUTPUT_DIR/$output_csv${NC}"
        fi

        echo ""
    done

    echo "============================================================"
    echo "Done!"
    echo "Success: $success_count, Failed: $fail_count, Skipped: $skip_count"
    echo "CUDA output directory: $CUDA_OUTPUT_DIR"
    echo "CPU  output directory: $CPU_OUTPUT_DIR"
    echo "============================================================"
}

# Run main function
main
