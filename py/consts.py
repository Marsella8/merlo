from pathlib import Path

REPO_ID = "QuantFactory/SmolLM2-135M-GGUF"
FILENAME = "SmolLM2-135M.Q8_0.gguf"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / FILENAME