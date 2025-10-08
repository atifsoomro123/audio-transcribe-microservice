# tests/test_api.py
from fastapi.testclient import TestClient
from src.app import app
import io

client = TestClient(app)

def test_health_and_transcribe_invalid():
    # invalid (no file)
    r = client.post("/v1/transcribe")
    assert r.status_code == 422

def test_transcribe_small_sample():
    # This test assumes you have sample audio at sample_audio/speech_clean.wav
    with open("sample_audio/speech_clean.wav", "rb") as fh:
        files = {"file": ("speech_clean.wav", fh, "audio/wav")}
        r = client.post("/v1/transcribe", files=files, data={"config": '{"model_size":"tiny","enable_separation":false}'})
        assert r.status_code == 200
        j = r.json()
        assert "request_id" in j
        assert "text" in j
