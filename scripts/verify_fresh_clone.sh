#!/bin/bash
# Fresh Clone Verification Script
# This script verifies that a fresh clone of the repository can be set up and tested

set -e  # Exit on error

echo "=========================================="
echo "Fresh Clone Verification Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        exit 1
    fi
}

# Check Python version
echo "Checking Python version..."
python3 --version > /dev/null 2>&1
print_status $? "Python 3 is available"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Warning: venv directory already exists, removing it...${NC}"
    rm -rf venv
fi
python3 -m venv venv
print_status $? "Virtual environment created"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_status $? "Virtual environment activated"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
pip install -r backend/requirements.txt > /dev/null 2>&1
print_status $? "Python dependencies installed"

# Run ruff linting
echo ""
echo "Running ruff linting..."
ruff check backend/ dspy_module/ tests/ --output-format=github > /dev/null 2>&1
print_status $? "Ruff linting passed"

# Run ruff format check
echo ""
echo "Running ruff format check..."
ruff format --check backend/ dspy_module/ tests/ > /dev/null 2>&1
print_status $? "Ruff format check passed"

# Run pytest tests
echo ""
echo "Running pytest tests..."
pytest tests/ -v --tb=short > /dev/null 2>&1
print_status $? "Pytest tests passed"

# Test backend imports
echo ""
echo "Testing backend imports..."
python -c "import backend.server; print('Backend imports OK')" > /dev/null 2>&1
print_status $? "Backend server imports OK"

python -c "from backend.rag_engine import get_rag_engine; print('RAG engine imports OK')" > /dev/null 2>&1
print_status $? "RAG engine imports OK"

# Test health endpoint with FastAPI TestClient
echo ""
echo "Testing health endpoint (FastAPI TestClient)..."
pytest tests/test_api.py::test_health_endpoint -v > /dev/null 2>&1
print_status $? "Health endpoint test passed"

# Check Node.js version
echo ""
echo "Checking Node.js version..."
if command -v node &> /dev/null; then
    node --version > /dev/null 2>&1
    print_status $? "Node.js is available"
else
    echo -e "${YELLOW}Warning: Node.js not found, skipping frontend tests${NC}"
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Backend verification completed successfully!${NC}"
    echo "=========================================="
    exit 0
fi

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd webapp
npm ci > /dev/null 2>&1
print_status $? "Frontend dependencies installed"

# Run frontend lint
echo ""
echo "Running frontend lint..."
npm run lint > /dev/null 2>&1
print_status $? "Frontend lint passed"

# Build frontend
echo ""
echo "Building frontend..."
npm run build > /dev/null 2>&1
print_status $? "Frontend build successful"

# Check build output
echo ""
echo "Checking build output..."
if [ -d ".next" ]; then
    print_status 0 "Build output directory exists"
else
    print_status 1 "Build output directory not found"
fi

cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}All verifications passed successfully!${NC}"
echo "=========================================="
echo ""
echo "The repository is ready for development."
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"

