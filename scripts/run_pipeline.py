"""
scripts/run_pipeline.py
-----------------------
Cross-platform pipeline automation script.
Runs notebooks 01 and 02 in order using nbconvert.
Works on Windows, Mac, and Linux.

Usage:
    python scripts/run_pipeline.py

Prerequisites:
    - Docker containers running (docker compose up -d postgres mongo)
    - Virtual environment activated
    - .env populated with GROQ_API_KEY and ALPHA_VANTAGE_KEY
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def run(cmd, description):
    print(f"\n{'='*60}\n  {description}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n  ERROR: {description} failed")
        return False
    print(f"\n  Done: {description}")
    return True


def main():
    print(f"\nFinancialSignalAgent — Pipeline Runner")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    root = Path(__file__).parent.parent
    os.chdir(root)

    notebooks = [
        ("notebooks/01_data_ingestion.ipynb", "Data Ingestion"),
        ("notebooks/02_pipeline.ipynb",        "Transform + Embed + Lineage"),
    ]

    for nb_path, label in notebooks:
        success = run([
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=600",
            nb_path
        ], label)

        if not success:
            print(f"\nStopped at: {nb_path}")
            sys.exit(1)

    print("\n" + "="*60)
    print(f"  Pipeline complete — {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)
    print("\nNext: open notebooks/03_agent_demo.ipynb")
    print("Then: uvicorn api.app:app --reload --port 8000")


if __name__ == "__main__":
    main()