# Verification Guide

This document provides commands to verify that the HR FAQ Chatbot project works correctly from a clean state.

## Prerequisites Check

```bash
# Check Python version (should be 3.8+)
python --version
# or
python3 --version

# Check Node.js version (should be 18+)
node --version

# Check npm
npm --version
```

## One-Command Startup Verification

### Linux/Mac

```bash
# Make script executable (if needed)
chmod +x start.sh

# Run startup script
./start.sh
```

**Expected output:**
- Virtual environment created (if missing)
- Dependencies installed
- Backend starts on port 8000
- Frontend starts on port 3000
- Health check passes
- URLs printed

**Verify:**
```bash
# In another terminal, check health
curl http://localhost:8000/health

# Should return JSON with status: "healthy"
```

### Windows

```powershell
# Run startup script
.\start.ps1
```

**Expected output:**
- Virtual environment created (if missing)
- Dependencies installed
- Backend starts on port 8000
- Frontend starts on port 3000
- Health check passes
- URLs printed

**Verify:**
```powershell
# In another PowerShell window
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing
```

## Manual Startup Verification

### Step 1: Backend

```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start backend
cd backend
uvicorn server:app --reload --port 8000
```

**Verify backend:**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test ask endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
```

### Step 2: Frontend

```bash
# In a new terminal
cd webapp

# Install dependencies (first time only)
npm install

# Start frontend
npm run dev
```

**Verify frontend:**
- Open http://localhost:3000 in browser
- Should see the HR FAQ Copilot interface
- Try asking a question
- Should receive a response

## End-to-End Test

1. **Start both services** (using `start.sh`/`start.ps1` or manually)

2. **Test frontend → backend communication:**
   - Open http://localhost:3000
   - Type: "How many vacation days do I get per year?"
   - Click Send or press Enter
   - Should see streaming response

3. **Test error handling:**
   - Stop backend
   - Try asking a question in frontend
   - Should see error message about backend being unreachable

4. **Test OOD detection:**
   - Ask: "What is the capital of France?"
   - Should be rejected as out-of-domain

## Benchmark Verification

```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run benchmark
python benchmark_professional.py
```

**Expected output:**
- Benchmark runs successfully
- Results saved to `reports/latest_run/`:
  - `metrics.json`
  - `predictions.jsonl`
  - `config.yaml` (or `config.json`)
  - `report.md`
  - `professional_benchmark_results.json`

**Verify files exist:**
```bash
ls reports/latest_run/
# Should show all output files
```

## CI Verification

The CI pipeline runs automatically on push/PR. To verify locally:

### Backend CI Steps

```bash
# Install CI dependencies
pip install pytest pytest-cov black flake8

# Lint
flake8 backend/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Format check
black --check backend/ --line-length 127

# Import test
python -c "import backend.server; print('OK')"
```

### Frontend CI Steps

```bash
cd webapp

# Install
npm ci

# Lint
npm run lint

# Build
npm run build

# Verify build
test -d .next && echo "Build OK" || echo "Build failed"
```

## Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is in use
# Linux/Mac
lsof -i :8000
# Windows
netstat -ano | findstr :8000

# Check Python dependencies
pip list | grep -E "fastapi|uvicorn"
```

### Frontend won't start

```bash
# Check if port 3000 is in use
# Linux/Mac
lsof -i :3000
# Windows
netstat -ano | findstr :3000

# Clear Next.js cache
cd webapp
rm -rf .next
npm run dev
```

### Frontend can't reach backend

```bash
# Check backend is running
curl http://localhost:8000/health

# Check CORS (should be configured in backend/server.py)
# Check NEXT_PUBLIC_API_URL in webapp/.env.local
```

### Benchmark fails

```bash
# Check model files exist
ls models/hr_faq_dialogpt_lora/

# Check data files
ls data/train_alpaca.json

# Run with verbose output
python benchmark_professional.py 2>&1 | tee benchmark.log
```

## Success Criteria

✅ One-command startup works (`./start.sh` or `.\start.ps1`)  
✅ Backend health check returns `{"status": "healthy"}`  
✅ Frontend loads at http://localhost:3000  
✅ Can ask questions and receive answers  
✅ Benchmark generates `reports/latest_run/*` files  
✅ CI pipeline passes (on GitHub Actions)  

## Quick Verification Script

```bash
#!/bin/bash
# quick-verify.sh

echo "🔍 Quick Verification..."

# Check Python
python --version || { echo "❌ Python not found"; exit 1; }

# Check Node
node --version || { echo "❌ Node.js not found"; exit 1; }

# Check backend health (if running)
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend not running (start with ./start.sh)"
fi

# Check frontend (if running)
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is running"
else
    echo "⚠️  Frontend not running (start with ./start.sh)"
fi

# Check benchmark output
if [ -d "reports/latest_run" ] && [ -f "reports/latest_run/metrics.json" ]; then
    echo "✅ Benchmark output exists"
else
    echo "⚠️  Run benchmark: python benchmark_professional.py"
fi

echo "✅ Verification complete"
```

