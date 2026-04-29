#!/usr/bin/env bash
# Quick verification that the fixes work

cd /run/media/warren/Education/UoM/FYP/AI-Framework

echo "Starting RF chain worker..."
python -m ai_framework.inference.rf_chain_worker --socket-path /tmp/verify_rfchain.sock > /tmp/verify_rf.log 2>&1 &
RF_PID=$!
sleep 2

echo "Starting inference worker..."
python -m ai_framework.inference.worker --socket-path /tmp/verify_infer.sock --checkpoint ./checkpoints/best_model.pt --scalers ./checkpoints/scalers.joblib --device cpu --sample-rate-hz 25000000 --shm-create > /tmp/verify_worker.log 2>&1 &
WORKER_PID=$!
sleep 2

echo "Running digital-twin for 5 cycles..."
rm -f inference_results.txt ai_inference.txt
cd /run/media/warren/Education/UoM/FYP/AI-Framework/software_framework
timeout 8 cargo run --release -- \
  --mode digital_twin \
  --ipc-mode shm \
  --socket-path /tmp/verify_infer.sock \
  --rf-chain-socket-path /tmp/verify_rfchain.sock \
  --sample-rate-hz 25000000 \
  --shm-name maars_iq_verify \
  --shm-slots 8 \
  --shm-slot-capacity 8192 \
  --rf-chain-cycles 5 \
  --rf-chain-interval-ms 100 \
  --enable-inference \
  --print-inference-results \
  2>&1 | grep -E "Seq:|digital_twin|agent_inference" || true

cd /run/media/warren/Education/UoM/FYP/AI-Framework
sleep 1

echo ""
echo "======= VERIFICATION RESULTS ======="
echo ""
echo "1. Snapshot file (seq_id should increment):"
if [ -f inference_results.txt ]; then
  grep "seq_id=" inference_results.txt | head -1
else
  echo "  [NO FILE]"
fi

echo ""
echo "2. AI Inference Log (should show agent decisions):"
if [ -f ai_inference.txt ]; then
  tail -3 ai_inference.txt
else
  echo "  [NO LOG]"
fi

echo ""
echo "Cleaning up..."
kill $RF_PID $WORKER_PID 2>/dev/null || true
rm -f /tmp/verify*.sock
echo "Done."
