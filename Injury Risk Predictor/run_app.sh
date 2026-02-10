#!/bin/bash
# Run both the FastAPI backend and Next.js frontend

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================="
echo "EPL Injury Risk Predictor"
echo "=================================="
echo ""

# Check if models exist
if [ ! -f "$PROJECT_DIR/models/stacking_ensemble.pkl" ]; then
    echo "WARNING: No trained models found in models/"
    echo "Run the training notebook first, or run:"
    echo "  python scripts/build_epl_predictions.py"
    echo ""
fi

# Start FastAPI backend
echo "Starting API server on http://localhost:8000..."
cd "$PROJECT_DIR"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 2

# Start Next.js frontend
echo "Starting frontend on http://localhost:3000..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=================================="
echo "App is running!"
echo "  Frontend: http://localhost:3000"
echo "  API:      http://localhost:8000"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap to kill both processes on exit
trap "kill $API_PID $FRONTEND_PID 2>/dev/null" EXIT

# Wait for either process to exit
wait
