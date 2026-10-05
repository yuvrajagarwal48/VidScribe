"""
VidScribe Main Launcher
-----------------------
Single command entry point to launch the VidScribe Web Application & API Server.

Usage:
    python run.py             # Start the Web UI & API server on http://127.0.0.1:8000
    python run.py --host 0.0.0.0 --port 8000  # For Docker / production binding
"""

import sys
import argparse
import uvicorn
import config

# Prevent UnicodeEncodeError on Windows console (cp1252) when printing fullwidth characters or emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="VidScribe Agentic VideoRAG Launcher")
    parser.add_argument("--host", default=config.SERVER_HOST, help="Host address to bind the server")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload on code change")

    args = parser.parse_args()

    print("=================================================================")
    print("          [*] Starting VidScribe Agentic VideoRAG Web UI          ")
    print(f"               URL: http://{args.host}:{args.port}              ")
    print(f"               LLM Provider: {config.LLM_PROVIDER}              ")
    print("=================================================================")
    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
