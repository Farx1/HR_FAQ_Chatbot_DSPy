# One-command startup script for HR FAQ Chatbot (Windows)
# Starts backend and frontend concurrently

Write-Host "🚀 Starting HR FAQ Chatbot..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3.8+ is required. Please install Python." -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js 18+ is required. Please install Node.js." -ForegroundColor Red
    exit 1
}

# Create venv if missing
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -r backend\requirements.txt

# Install webapp dependencies
if (-not (Test-Path "webapp\node_modules")) {
    Write-Host "📦 Installing webapp dependencies..." -ForegroundColor Yellow
    Set-Location webapp
    npm install --silent
    Set-Location ..
}

# Start backend in background
Write-Host "🔧 Starting backend server (port 8000)..." -ForegroundColor Yellow
Set-Location backend
Start-Process python -ArgumentList "-m", "uvicorn", "server:app", "--reload", "--port", "8000", "--host", "0.0.0.0" -WindowStyle Hidden -RedirectStandardOutput "..\backend.log" -RedirectStandardError "..\backend.log"
Set-Location ..

# Wait for backend to be ready
Write-Host "⏳ Waiting for backend to be ready..." -ForegroundColor Yellow
$backendReady = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Backend is ready!" -ForegroundColor Green
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $backendReady) {
    Write-Host "❌ Backend failed to start. Check backend.log" -ForegroundColor Red
    exit 1
}

# Start frontend
Write-Host "🎨 Starting frontend (port 3000)..." -ForegroundColor Yellow
Set-Location webapp
Start-Process npm -ArgumentList "run", "dev" -WindowStyle Hidden -RedirectStandardOutput "..\frontend.log" -RedirectStandardError "..\frontend.log"
Set-Location ..

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "✅ HR FAQ Chatbot is running!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 URLs:" -ForegroundColor Cyan
Write-Host "   Backend:  http://localhost:8000"
Write-Host "   Frontend: http://localhost:3000"
Write-Host "   Health:   http://localhost:8000/health"
Write-Host ""
Write-Host "📝 Logs:" -ForegroundColor Cyan
Write-Host "   Backend:  Get-Content backend.log -Wait"
Write-Host "   Frontend: Get-Content frontend.log -Wait"
Write-Host ""
Write-Host "🛑 To stop: Close this window or press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Keep script running
Read-Host "Press Enter to stop servers"

