"""
DataForge Configuration Module
-----------------------------
Centralized configuration management for paths, model providers, 
hyperparameters, and server settings. Reads environment variables from .env.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure Python environment Scripts directory (containing ffmpeg.exe) is in PATH
scripts_dir = os.path.dirname(sys.executable)
if scripts_dir and scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

# Prevent UnicodeEncodeError on Windows consoles when handling unicode filenames or emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. Model Configuration (Google Gemini)
# ==========================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ==========================================
# 2. Directory Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"
KEYFRAMES_DIR = PROCESSED_DIR / "keyframes"
VECTOR_DB_DIR = PROCESSED_DIR / "vectordb"
SUMMARIES_DIR = BASE_DIR / "summaries"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure all vital directories exist at startup
for directory in [DATA_DIR, PROCESSED_DIR, KEYFRAMES_DIR, VECTOR_DB_DIR, SUMMARIES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. Media Processing Hyperparameters
# ==========================================
# Scene Detection Threshold (ContentDetector in PySceneDetect)
# Higher value = fewer scenes detected (lower sensitivity)
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "20.0"))
MIN_SCENE_LEN = int(os.getenv("MIN_SCENE_LEN", "15"))  # Minimum frames per scene

# Tesseract OCR Confidence Threshold (0 - 100)
OCR_CONF_THRESHOLD = int(os.getenv("OCR_CONF_THRESHOLD", "60"))

# Whisper model name ("tiny", "base", "small", "medium", "large")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ==========================================
# 4. Vector Store & RAG Parameters
# ==========================================
VECTOR_DB_COLLECTION = "video_multimodal_rag"
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Embedding provider ("auto", "gemini" for 0MB local RAM, or "sentence_transformers")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto").lower()

# ==========================================
# 5. Web Server Configuration
# ==========================================
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))