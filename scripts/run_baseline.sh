
### 9. scripts/run_baseline.sh (Updated)

```bash
#!/bin/bash

# Fraud Investigation Baseline Runner
# Uses Meta Llama 3.1 8B via free HF Inference API

export ENV_URL="${ENV_URL:-http://localhost:7860}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api-inference.huggingface.co/v1}"

echo "Environment: $ENV_URL"
echo "Model: $MODEL_NAME"
echo "API: $OPENAI_BASE_URL"

if [ -z "$OPENAI_API_KEY" ] && [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: No OPENAI_API_KEY or HF_TOKEN set."
    echo "Running with rule-based fallback agent."
    echo ""
    echo "To use Meta Llama 3.1 8B (free):"
    echo "  export OPENAI_API_KEY=hf_your_token_here"
    echo ""
fi

# Use HF_TOKEN as OPENAI_API_KEY if only HF_TOKEN is set
if [ -z "$OPENAI_API_KEY" ] && [ -n "$HF_TOKEN" ]; then
    export OPENAI_API_KEY="$HF_TOKEN"
fi

python baseline/inference.py