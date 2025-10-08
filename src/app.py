# src/app.py
import os
import uuid
import json
import shutil
import time
import logging
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pipeline import process_audio_file
from utils import setup_logging

setup_logging()
logger = logging.getLogger("transcribe_api")

app = FastAPI(title="Audio Transcription + Separation API",
              version="0.1.0",
              description="POST /v1/transcribe with multipart/form-data `file` and optional `config` JSON.")

class ConfigModel(BaseModel):
    language_hint: Optional[str] = None
    enable_separation: Optional[bool] = True
    diarize: Optional[bool] = False
    model_size: Optional[str] = "small"
    target_sr: Optional[int] = 16000

@app.post("/v1/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None)
):
    """
    POST /v1/transcribe
      - file: audio binary (wav/mp3/m4a/flac/ogg)
      - config: JSON string (language_hint, enable_separation, diarize, model_size, target_sr)
    """
    request_id = str(uuid.uuid4())
    start_total = time.time()
    logger.info(f"[{request_id}] Received request: filename={file.filename} content_type={file.content_type}")

    # parse config JSON if provided
    try:
        cfg = ConfigModel(**(json.loads(config) if config else {}))
    except Exception as e:
        logger.exception("Invalid config JSON")
        raise HTTPException(status_code=400, detail="Invalid config JSON")

    # Basic file-size guard (return 413 if > 200 MB)
    uploaded_bytes = await file.read()
    max_bytes = 200 * 1024 * 1024
    if len(uploaded_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large (max 200MB)")

    tmpdir = tempfile.mkdtemp(prefix=f"req_{request_id}_")
    try:
        input_path = os.path.join(tmpdir, file.filename)
        with open(input_path, "wb") as fh:
            fh.write(uploaded_bytes)
        # process pipeline
        result = process_audio_file(
            input_path=input_path,
            request_id=request_id,
            language_hint=cfg.language_hint,
            enable_separation=cfg.enable_separation,
            diarize=cfg.diarize,
            model_size=cfg.model_size,
            target_sr=cfg.target_sr,
            tmpdir=tmpdir
        )
        total_ms = int((time.time() - start_total) * 1000)
        result["timings_ms"]["total"] = total_ms
        result["request_id"] = request_id
        logger.info(f"[{request_id}] Completed request in {total_ms}ms")
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error")
        raise HTTPException(status_code=500, detail={"request_id": request_id, "error": str(e)})
    finally:
        # optionally cleanup tmpdir here; keep it for debugging if you prefer
        shutil.rmtree(tmpdir, ignore_errors=True)
