PYTHON = python3
PIP = pip
LLM_MODEL = llama3

install:
	$(PIP) install -r requirements.txt
	@echo ">>> Python dependencies installed."
	@echo ">>> Pulling Ollama model: $(LLM_MODEL)..."
	ollama pull $(LLM_MODEL)

# Run the default script
run_stock:
	python sp_500.py

run_crypto:
	python crypto_trading.py

clean:
	rm -rf __pycache__