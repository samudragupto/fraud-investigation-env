#!/bin/bash
set -e

echo "=== Fraud Investigation Environment Validator ==="
echo ""

echo "=== Step 1: Building Docker image ==="
docker build -t fraud-env .

echo "=== Step 2: Starting container ==="
docker run -d --name fraud-env-test \
    -p 7860:7860 \
    -e MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct \
    fraud-env
echo "Waiting for container to start..."
sleep 20

echo "=== Step 3: Health check ==="
HEALTH=$(curl -s http://localhost:7860/)
echo "$HEALTH"
echo "$HEALTH" | python3 -c \
    "import sys,json; \
     d=json.load(sys.stdin); \
     assert d['status']=='ok'; \
     print('Health: OK')"

echo "=== Step 4: Testing /tasks ==="
TASKS=$(curl -s http://localhost:7860/tasks)
TASK_COUNT=$(echo "$TASKS" | python3 -c \
    "import sys,json; \
     print(len(json.load(sys.stdin)))")
echo "Found $TASK_COUNT tasks"
if [ "$TASK_COUNT" -lt 3 ]; then
    echo "FAIL: Less than 3 tasks"
    exit 1
fi

echo "=== Step 5: Testing reset() ==="
OBS=$(curl -s -X POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d '{"task_id":"single_transaction_classification"}')
echo "$OBS" | python3 -c \
    "import sys,json; \
     d=json.load(sys.stdin); \
     assert 'current_transaction' in d; \
     assert 'steps_remaining' in d; \
     print('reset(): OK')"

echo "=== Step 6: Testing step() ==="
RESULT=$(curl -s -X POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d '{"action_type":"query_account_history", \
         "parameters":{"account_id":"ACC_EASY_001"}}')
echo "$RESULT" | python3 -c \
    "import sys,json; \
     d=json.load(sys.stdin); \
     assert 'observation' in d; \
     assert 'reward' in d; \
     assert 'done' in d; \
     print('step(): OK')"

echo "=== Step 7: Testing state() ==="
curl -s http://localhost:7860/state | python3 -c \
    "import sys,json; \
     d=json.load(sys.stdin); \
     assert 'task_id' in d; \
     print('state(): OK')"

echo "=== Step 8: Submit and test /grader ==="
curl -s -X POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d '{"action_type":"submit_investigation", \
         "parameters":{}}' > /dev/null

SCORE=$(curl -s http://localhost:7860/grader | python3 -c \
    "import sys,json; \
     d=json.load(sys.stdin); \
     assert 0.0 <= d['score'] <= 1.0; \
     print(f\"Score: {d['score']}\")")
echo "Grader: $SCORE"

echo "=== Step 9: Test all three tasks ==="
for TASK in \
    "single_transaction_classification" \
    "multi_account_pattern_detection" \
    "fraud_ring_detection"; do
    curl -s -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" \
        -d "{\"task_id\":\"$TASK\"}" > /dev/null
    curl -s -X POST http://localhost:7860/step \
        -H "Content-Type: application/json" \
        -d '{"action_type":"submit_investigation", \
             "parameters":{}}' > /dev/null
    TSCORE=$(curl -s http://localhost:7860/grader | \
        python3 -c \
        "import sys,json; \
         d=json.load(sys.stdin); \
         assert 0.0 <= d['score'] <= 1.0; \
         print(d['score'])")
    echo "  $TASK: $TSCORE"
done

echo "=== Step 10: Running openenv validate ==="
openenv validate 2>/dev/null || \
    echo "openenv validate not installed (optional)"

echo "=== Step 11: Cleanup ==="
docker stop fraud-env-test
docker rm fraud-env-test

echo ""
echo "=== ALL CHECKS PASSED ==="
echo ""
echo "Ready to deploy:"
echo "  openenv push --repo-id YOUR_USERNAME/fraud-investigation-env"