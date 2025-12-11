PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python
NB_CONVERT := $(VENV)/bin/jupyter
REQ := requirements.txt
NOTEBOOK := training_notebook.ipynb
BOT := patchtst_alpaca_bot.py

.PHONY: all venv install requirements train run run-live clean show

all: venv install run

venv:
	@echo "==> Creating virtualenv at $(VENV)"
	@test -d $(VENV) || $(PY) -m venv $(VENV)
	@$(PIP) --version >/dev/null 2>&1 || (echo "pip not available in venv"; exit 1)

requirements: $(REQ)

$(REQ):
	@echo "==> Writing minimal $(REQ)"
	@cat > $(REQ) << 'EOF'
torch
pandas
numpy
scikit-learn
matplotlib
pytz
alpaca-py
jupyter
nbconvert
EOF
	@echo "$(REQ) written"

install: venv requirements
	@echo "==> Installing dependencies into $(VENV)"
	@$(PIP) install --upgrade pip setuptools wheel >/dev/null
	@$(PIP) install -r $(REQ)

# Execute training notebook in-place (will run all cells)
train: install
	@echo "==> Executing notebook: $(NOTEBOOK)"
	@$(PYBIN) -m nbconvert --to notebook --execute $(NOTEBOOK) --inplace

# Run the trading bot in DRY RUN mode (the script's default DRY_RUN=True)
run: install
	@echo "==> Running bot (dry-run): $(BOT)"
	@$(PYBIN) $(BOT)

# Run the trading bot in LIVE mode (will attempt to place orders).
# NOTE: The script's internal DRY_RUN default may still prevent live orders;
# ensure you understand the script and set DRY_RUN=False inside it before using this.
run-live: install
	@echo "==> RUNNING BOT IN LIVE MODE - make sure you understand this will place orders in your paper account"
	@$(PYBIN) $(BOT)

# Print quick info / sanity checks
show:
	@echo "Project files:"
	@ls -1 --color=never models 2>/dev/null || true
	@echo " - Notebook: $(NOTEBOOK)"
	@echo " - Bot: $(BOT)"
	@echo " - Requirements: $(REQ)"
	@echo "Virtualenv: $(VENV)"

clean:
	@echo "==> Cleaning virtualenv, caches, and python artifacts"
	@rm -rf $(VENV) __pycache__ .pytest_cache build dist *.egg-info
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@echo "cleaned"

# convenience: rebuild env from scratch
rebuild: clean venv install