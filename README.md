# Audio Transcribe Microservice (Demucs + Whisper)

## What this project provides
A containerized FastAPI microservice that:
- accepts an audio file (`wav/mp3/m4a/flac/ogg`) via `POST /v1/transcribe`
- optionally runs source separation (Demucs) to isolate vocals
- falls back to noise-reduction if separation fails
- runs ASR (OpenAI Whisper) on separated vocals
- returns structured JSON with segments, text, language, request_id, and timings

## API
`POST /v1/transcribe` (multipart/form-data)
- `file`: binary audio file
- `config` (optional): JSON string, e.g.
  ```json
  {
    "language_hint": "en",
    "enable_separation": true,
    "diarize": false,
    "model_size": "small",
    "target_sr": 16000
  }
