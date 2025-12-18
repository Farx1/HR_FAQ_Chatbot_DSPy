.PHONY: dev install clean test lint

# Default target
.DEFAULT_GOAL := dev

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Python and Node versions
PYTHON := python3
NODE := node
NPM := npm

# Directories
VENV := venv
BACKEND_DIR := backend
WEBAPP_DIR := webapp

dev: install
	@echo "$(YELLOW)🚀 Starting HR FAQ Chatbot...$(NC)"
	@echo ""
	@echo "$(YELLOW)📦 Starting backend on port 8000...$(NC)"
	@cd $(BACKEND_DIR) && $(VENV)/bin/python -m uvicorn server:app --reload --port 8000 --host 0.0.0.0 > ../backend.log 2>&1 & echo $$! > ../backend.pid
	@sleep 3
	@echo "$(YELLOW)📦 Starting frontend on port 3000...$(NC)"
	@cd $(WEBAPP_DIR) && $(NPM) run dev > ../frontend.log 2>&1 & echo $$! > ../frontend.pid
	@sleep 5
	@echo ""
	@echo "$(GREEN)✅ HR FAQ Chatbot is running!$(NC)"
	@echo ""
	@echo "$(GREEN)📍 URLs:$(NC)"
	@echo "   Backend:  http://localhost:8000"
	@echo "   Frontend: http://localhost:3000"
	@echo "   Health:   http://localhost:8000/health"
	@echo ""
	@echo "$(YELLOW)📝 Logs:$(NC)"
	@echo "   Backend:  tail -f backend.log"
	@echo "   Frontend: tail -f frontend.log"
	@echo ""
	@echo "$(YELLOW)🛑 To stop: make stop$(NC)"
	@echo ""

install: venv python-deps node-deps
	@echo "$(GREEN)✅ All dependencies installed$(NC)"

venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)📦 Creating Python virtual environment...$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
	fi

python-deps: venv
	@echo "$(YELLOW)📦 Installing Python dependencies...$(NC)"
	@$(VENV)/bin/pip install -q --upgrade pip
	@$(VENV)/bin/pip install -q -r requirements.txt
	@$(VENV)/bin/pip install -q -r $(BACKEND_DIR)/requirements.txt

node-deps:
	@if [ ! -d "$(WEBAPP_DIR)/node_modules" ]; then \
		echo "$(YELLOW)📦 Installing webapp dependencies...$(NC)"; \
		cd $(WEBAPP_DIR) && $(NPM) install --silent; \
	fi

stop:
	@if [ -f backend.pid ]; then \
		kill $$(cat backend.pid) 2>/dev/null || true; \
		rm backend.pid; \
		echo "$(GREEN)✅ Backend stopped$(NC)"; \
	fi
	@if [ -f frontend.pid ]; then \
		kill $$(cat frontend.pid) 2>/dev/null || true; \
		rm frontend.pid; \
		echo "$(GREEN)✅ Frontend stopped$(NC)"; \
	fi

test:
	@echo "$(YELLOW)🧪 Running tests...$(NC)"
	@$(VENV)/bin/pytest tests/ -v

lint:
	@echo "$(YELLOW)🔍 Running linter...$(NC)"
	@$(VENV)/bin/ruff check backend/ dspy_module/

clean:
	@echo "$(YELLOW)🧹 Cleaning up...$(NC)"
	@make stop || true
	@rm -f backend.log frontend.log backend.pid frontend.pid
	@rm -rf $(VENV)
	@rm -rf $(WEBAPP_DIR)/node_modules
	@rm -rf $(WEBAPP_DIR)/.next

