# Define the Python interpreter to use. 
# This helps ensure consistency (e.g., using a virtual environment's Python).
PYTHON := python

# ----------------------------------------------------------------
# PHONY targets: Declares targets that don't produce files, 
# ensuring the commands are executed every time you call 'make'.
# ----------------------------------------------------------------
.PHONY: all train test clean

# Default target: Run both training and testing.
# Running 'make' will execute this target.
all: train test

# Target to run the training script (main.py)
train: ## Run the main training script
	@echo "--- Starting Training (main.py) ---"
	$(PYTHON) main.py

# Target to run the testing script (test.py)
test: ## Run the main testing script
	@echo "--- Starting Testing (test.py) ---"
	$(PYTHON) test.py

# Target to remove generated Python cache files
clean: ## Remove Python cache files (__pycache__ and *.pyc)
	@echo "--- Cleaning up temporary files ---"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Optional: Add a 'help' target for documentation
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'