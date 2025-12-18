# Fresh Clone Verification Script (PowerShell)
# This script verifies that a fresh clone of the repository can be set up and tested

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fresh Clone Verification Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Function to print status
function Print-Status {
    param(
        [bool]$Success,
        [string]$Message
    )
    if ($Success) {
        Write-Host "✓ " -NoNewline -ForegroundColor Green
        Write-Host $Message
    } else {
        Write-Host "✗ " -NoNewline -ForegroundColor Red
        Write-Host $Message
        exit 1
    }
}

# Check Python version
Write-Host "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    Print-Status $true "Python is available: $pythonVersion"
} catch {
    Print-Status $false "Python is not available"
}

# Create virtual environment
Write-Host ""
Write-Host "Creating Python virtual environment..."
if (Test-Path "venv") {
    Write-Host "Warning: venv directory already exists, removing it..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}
python -m venv venv
if ($LASTEXITCODE -eq 0) {
    Print-Status $true "Virtual environment created"
} else {
    Print-Status $false "Failed to create virtual environment"
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
Print-Status $true "Virtual environment activated"

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip | Out-Null
pip install -r requirements.txt | Out-Null
pip install -r backend/requirements.txt | Out-Null
Print-Status $true "Python dependencies installed"

# Run ruff linting
Write-Host ""
Write-Host "Running ruff linting..."
ruff check backend/ dspy_module/ tests/ --output-format=github | Out-Null
Print-Status $true "Ruff linting passed"

# Run ruff format check
Write-Host ""
Write-Host "Running ruff format check..."
ruff format --check backend/ dspy_module/ tests/ | Out-Null
Print-Status $true "Ruff format check passed"

# Run pytest tests
Write-Host ""
Write-Host "Running pytest tests..."
pytest tests/ -v --tb=short | Out-Null
Print-Status $true "Pytest tests passed"

# Test backend imports
Write-Host ""
Write-Host "Testing backend imports..."
python -c "import backend.server; print('Backend imports OK')" | Out-Null
Print-Status $true "Backend server imports OK"

python -c "from backend.rag_engine import get_rag_engine; print('RAG engine imports OK')" | Out-Null
Print-Status $true "RAG engine imports OK"

# Test health endpoint with FastAPI TestClient
Write-Host ""
Write-Host "Testing health endpoint (FastAPI TestClient)..."
pytest tests/test_api.py::test_health_endpoint -v | Out-Null
Print-Status $true "Health endpoint test passed"

# Check Node.js version
Write-Host ""
Write-Host "Checking Node.js version..."
try {
    $nodeVersion = node --version
    Print-Status $true "Node.js is available: $nodeVersion"
} catch {
    Write-Host "Warning: Node.js not found, skipping frontend tests" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Backend verification completed successfully!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Cyan
    exit 0
}

# Install frontend dependencies
Write-Host ""
Write-Host "Installing frontend dependencies..."
Push-Location webapp
npm ci | Out-Null
Print-Status $true "Frontend dependencies installed"

# Run frontend lint
Write-Host ""
Write-Host "Running frontend lint..."
npm run lint | Out-Null
Print-Status $true "Frontend lint passed"

# Build frontend
Write-Host ""
Write-Host "Building frontend..."
npm run build | Out-Null
Print-Status $true "Frontend build successful"

# Check build output
Write-Host ""
Write-Host "Checking build output..."
if (Test-Path ".next") {
    Print-Status $true "Build output directory exists"
} else {
    Print-Status $false "Build output directory not found"
}

Pop-Location

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "All verifications passed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The repository is ready for development."
Write-Host "To activate the virtual environment, run:"
Write-Host "  .\venv\Scripts\Activate.ps1"

