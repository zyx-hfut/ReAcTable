#!/bin/bash
# Setup script for ReAcTable local inference with Qwen2.5-3B-Instruct via vLLM
#
# Usage:
#   bash local_inference/setup_env.sh          # Full setup (create env + install deps)
#   bash local_inference/setup_env.sh server   # Start vLLM server only
#   bash local_inference/setup_env.sh test     # Test vLLM connectivity only

set -e

ENV_NAME="reactable-qwen"
PYTHON_VERSION="3.9"
VLLM_PORT=8000
MODEL="Qwen/Qwen2.5-3B-Instruct"
SERVED_NAME="qwen2.5-3b"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

setup_conda() {
    echo "=== Creating conda environment: $ENV_NAME ==="
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    echo "=== Installing dependencies ==="
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    cd "$PROJECT_ROOT"
    pip install -e .
    pip install vllm

    echo "=== Environment setup complete ==="
    echo "Activate with: conda activate $ENV_NAME"
}

start_server() {
    echo "=== Starting vLLM server ==="
    echo "Model: $MODEL"
    echo "Served as: $SERVED_NAME"
    echo "Port: $VLLM_PORT"

    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --dtype auto
}

test_connectivity() {
    echo "=== Testing vLLM connectivity ==="
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:$VLLM_PORT/v1', api_key='EMPTY')

# List models
models = client.models.list()
print('Available models:', [m.id for m in models.data])

# Simple generation test
response = client.chat.completions.create(
    model='$SERVED_NAME',
    messages=[{'role': 'user', 'content': 'Output exactly: SQL: \`\`\`SELECT 1\`\`\`'}],
    max_tokens=32,
    temperature=0,
)
print('Test response:', response.choices[0].message.content)
print('Connectivity OK!')
"
}

case "${1:-setup}" in
    setup)
        setup_conda
        echo ""
        echo "Next steps:"
        echo "  1. Start vLLM server:  bash local_inference/setup_env.sh server"
        echo "  2. Test connectivity:   bash local_inference/setup_env.sh test"
        echo "  3. Run experiment:      conda activate $ENV_NAME && python local_inference/run_wikitq.py --limit 5"
        ;;
    server)
        start_server
        ;;
    test)
        test_connectivity
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 [setup|server|test]"
        exit 1
        ;;
esac
