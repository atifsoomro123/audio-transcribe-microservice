# src/cli.py
"""
Simple CLI utility to post an audio file to the API and print the transcript.
Usage:
    python -m src.cli /path/to/audio.wav
"""
import sys
import requests
import json

API_URL = "http://localhost:8000/v1/transcribe"

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli /path/to/audio [config.json]")
        sys.exit(1)
    path = sys.argv[1]
    config = None
    if len(sys.argv) >= 3:
        config = json.load(open(sys.argv[2], "r"))
    files = {"file": open(path, "rb")}
    data = {}
    if config:
        data["config"] = json.dumps(config)
    print("Uploading...")
    r = requests.post(API_URL, files=files, data=data, timeout=600)
    print("Status:", r.status_code)
    print("Response:")
    print(r.text)

if __name__ == "__main__":
    main()
