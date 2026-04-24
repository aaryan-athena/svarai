"""
Start the SvarAI web server.

Usage:
    python run.py                  # default: localhost:8000
    python run.py --port 8080
    python run.py --host 0.0.0.0   # expose on local network
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Start the SvarAI web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not found. Run:  pip install -r requirements.txt")
        sys.exit(1)

    print(f"\n  SvarAI is running at  http://{args.host}:{args.port}")
    print(f"  Admin panel           http://{args.host}:{args.port}/admin")
    print(f"  API docs              http://{args.host}:{args.port}/docs\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
