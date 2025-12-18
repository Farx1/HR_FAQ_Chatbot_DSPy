#!/bin/bash
# One-command startup script for HR FAQ Chatbot (Linux/Mac)
# Starts backend and frontend concurrently

set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Starting HR FAQ Chatbot...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "❌ Python 3.8+ is required. Please install Python."
    exit 1
fi
echo -e "${GREEN}✅ Found: $(python3 --version)${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "❌ Node.js 18+ is required. Please install Node.js."
    exit 1
fi
echo -e "${GREEN}✅ Found: $(node --version)${NC}"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r backend/requirements.txt

# Install webapp dependencies
if [ ! -d "webapp/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing webapp dependencies...${NC}"
    cd webapp
    npm install --silent
    cd ..
fi

# Start backend in background
echo -e "${YELLOW}🔧 Starting backend server (port 8000)...${NC}"
cd backend
python -m uvicorn server:app --reload --port 8000 --host 0.0.0.0 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "❌ Backend failed to start. Check backend.log"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Start frontend
echo -e "${YELLOW}🎨 Starting frontend (port 3000)...${NC}"
cd webapp
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 3

echo ""
echo -e "${GREEN}✅ HR FAQ Chatbot is running!${NC}"
echo ""
echo -e "${GREEN}📍 URLs:${NC}"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo "   Health:   http://localhost:8000/health"
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo -e "${YELLOW}🛑 To stop: kill $BACKEND_PID $FRONTEND_PID${NC}"
echo ""

# Keep script running
wait

