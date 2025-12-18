#!/bin/bash
# One-command startup script for HR FAQ Chatbot
# Starts backend and frontend concurrently

set -e

echo "🚀 Starting HR FAQ Chatbot..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.8+ is required. Please install Python."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 18+ is required. Please install Node.js."
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r backend/requirements.txt

# Install webapp dependencies
if [ ! -d "webapp/node_modules" ]; then
    echo "📦 Installing webapp dependencies..."
    cd webapp
    npm install --silent
    cd ..
fi

# Start backend in background
echo "🔧 Starting backend server (port 8000)..."
cd backend
python -m uvicorn server:app --reload --port 8000 --host 0.0.0.0 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Backend failed to start. Check backend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Start frontend
echo "🎨 Starting frontend (port 3000)..."
cd webapp
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait a bit for frontend
sleep 3

echo ""
echo "✅ HR FAQ Chatbot is running!"
echo ""
echo "📍 URLs:"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Health:   http://localhost:8000/health"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Keep script running
wait

