# src/pipeline.py
import os
import subprocess
import time
import logging
import glob
import math
from typing import Optional, Dict, List
import soundfile as sf
import numpy as np
import librosa
import uuid
import json
import tempfile
from noisereduce import reduce_noise

from whisper import load_model   # from openai-whisper package

logger = logging.getLogger("transcribe_api.pipeline")

# model cache for fast reuse across requests
_WHISPER_MODELS = {}

def timed():
    return int(time.time() * 1000)

def _ffmpeg_convert_to_wav(input_path: str, output_path: str, target_sr: int = 16000):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-ac", "1", "-ar", str(target_sr),
        output_path
    ]
    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    return output_path

def _read_wav_info(wav_path: str):
    data, sr = sf.read(wav_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # to mono
    duration = len(data) / sr
    return {"sample_rate": sr, "duration_sec": duration}

def _demucs_separate(input_wav: str, out_dir: str):
    """
    Use demucs CLI (installed via pip) to perform two-stem separation: vocals + rest.
    Returns path to vocals file if found, otherwise raises.
    """
    logger.info("Attempting separation with Demucs CLI")
    cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs_ft", "--mp3", "-o", out_dir, input_wav]
    # note: model name 'htdemucs_ft' is a good default; adjust as needed.
    subprocess.check_call(cmd)
    # The demucs CLI writes output like out_dir/<input_basename>/vocals.wav
    base = os.path.splitext(os.path.basename(input_wav))[0]
    search = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(".wav") and ("vocals" in f.lower() or "vocals" in root.lower()):
                search.append(os.path.join(root, f))
    if not search:
        raise FileNotFoundError("Demucs produced no vocals file")
    # pick the first matched file
    return search[0]

def _noise_reduction_fallback(input_wav: str, output_wav: str):
    """
    Simple spectral gating noise reduction using noisereduce.
    """
    logger.info("Using noise-reduction fallback (noisereduce)")
    y, sr = librosa.load(input_wav, sr=None, mono=True)
    # Estimate noise from first 0.5s (if available)
    noise_clip = y[: min(len(y), int(0.5 * sr))]
    reduced = reduce_noise(y=y, sr=sr, y_noise=noise_clip)
    sf.write(output_wav, reduced, sr)
    return output_wav

def _load_whisper_model(size: str):
    if size in _WHISPER_MODELS:
        return _WHISPER_MODELS[size]
    logger.info("Loading Whisper model size=%s", size)
    model = load_model(size)
    _WHISPER_MODELS[size] = model
    return model

def _transcribe_whisper(model_size: str, audio_path: str, language: Optional[str] = None):
    start = timed()
    model = _load_whisper_model(model_size)
    options = {}
    if language:
        options["language"] = language
    # For CPU use fp16=False; whisper API will detect
    result = model.transcribe(audio_path, **options)
    timings = {"transcription_ms": int(timed() - start)}
    return result, timings

def _chunk_audio_and_transcribe(wav_path: str, model_size: str, language: Optional[str], chunk_length_sec:int=30, overlap_sec:int=5):
    """
    If the audio is long, split into overlapping chunks and transcribe each chunk then stitch.
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    total_dur = len(y) / sr
    chunks = []
    step = chunk_length_sec - overlap_sec
    n_chunks = max(1, math.ceil((total_dur - overlap_sec) / step))
    logger.info("Chunking audio: duration=%.1f sec, chunk_length=%d, overlap=%d -> %d chunks",
                total_dur, chunk_length_sec, overlap_sec, n_chunks)

    segments_all = []
    text_all = []
    timings = {"transcription_ms": 0}
    for i in range(n_chunks):
        start_time = max(0, i * step)
        end_time = min(total_dur, start_time + chunk_length_sec)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk = y[start_sample:end_sample]
        tmp_chunk_path = f"{wav_path}.chunk{i}.wav"
        sf.write(tmp_chunk_path, chunk, sr)
        res, t = _transcribe_whisper(model_size, tmp_chunk_path, language)
        timings["transcription_ms"] += t["transcription_ms"]
        # adjust times of returned segments
        for seg in res.get("segments", []):
            seg_adj = {
                "start": round(seg["start"] + start_time, 3),
                "end": round(seg["end"] + start_time, 3),
                "text": seg["text"].strip()
            }
            segments_all.append(seg_adj)
            text_all.append(seg["text"].strip())
        try:
            os.remove(tmp_chunk_path)
        except Exception:
            pass
    full_text = " ".join([t for t in text_all if t])
    return {"segments": segments_all, "text": full_text}, timings

def process_audio_file(input_path: str,
                       request_id: str,
                       language_hint: Optional[str] = None,
                       enable_separation: bool = True,
                       diarize: bool = False,
                       model_size: str = "small",
                       target_sr: int = 16000,
                       tmpdir: Optional[str] = None) -> Dict:
    timings = {"load": 0, "separation": 0, "transcription": 0, "total": 0}
    start_load = timed()
    base_tmp = tmpdir or tempfile.mkdtemp()
    # 1) convert to WAV 16k mono
    wav_path = os.path.join(base_tmp, f"{uuid.uuid4().hex}_normalized.wav")
    _ffmpeg_convert_to_wav(input_path, wav_path, target_sr)
    info = _read_wav_info(wav_path)
    timings["load"] = int(timed() - start_load)

    # 2) separation (attempt demucs)
    vocals_path = None
    separation_start = timed()
    separation_used = None
    try:
        if enable_separation:
            demucs_out = os.path.join(base_tmp, "demucs_out")
            os.makedirs(demucs_out, exist_ok=True)
            try:
                vocals_path = _demucs_separate(wav_path, demucs_out)
                separation_used = "demucs"
            except Exception as e:
                logger.exception("Demucs separation failed: %s", e)
                # fallback to noise reduction
                fallback_vocals = os.path.join(base_tmp, "vocals_nr.wav")
                vocals_path = _noise_reduction_fallback(wav_path, fallback_vocals)
                separation_used = "noise-reduction-fallback"
        else:
            separation_used = "disabled"
            vocals_path = wav_path
    except Exception as e:
        # If separation step raises, fallback gracefully
        logger.exception("Separation stage failed completely, falling back to original audio")
        vocals_path = wav_path
        separation_used = "failed-fallback-to-original"
    timings["separation"] = int(timed() - separation_start)

    # 3) If file is long, chunk then transcribe
    trans_start = timed()
    results = None
    try:
        if info["duration_sec"] > 45:
            # chunk and transcribe
            logger.info("Long file detected (%.2fs). Using chunking.", info["duration_sec"])
            trans, t = _chunk_audio_and_transcribe(vocals_path, model_size, language_hint,
                                                   chunk_length_sec=30, overlap_sec=5)
            timings["transcription"] = t["transcription_ms"]
            results = trans
        else:
            # single-shot transcribe
            res, t = _transcribe_whisper(model_size, vocals_path, language_hint)
            timings["transcription"] = t["transcription_ms"]
            # res contains 'text' and 'segments'
            segments = []
            for seg in res.get("segments", []):
                segments.append({
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "text": seg["text"].strip()
                })
            results = {
                "segments": segments,
                "text": res.get("text", "").strip(),
                "language": res.get("language", language_hint)
            }
    except Exception as e:
        logger.exception("Transcription failed")
        raise

    timings["transcription"] = int(timed() - trans_start)

    # 4) diarization (OPTIONAL) - placeholder (would integrate pyannote.audio here)
    diarization_result = None
    if diarize:
        # For now return a flag; a production implementation would load pyannote pipeline
        logger.warning("Diarization requested but not implemented in this reference. Set diarize=False.")
        diarization_result = {"warning": "diarization not implemented in reference code"}

    # Compose response
    response = {
        "duration_sec": round(info["duration_sec"], 3),
        "sample_rate": int(info["sample_rate"]),
        "pipeline": {
            "separation": {"enabled": enable_separation, "method": separation_used},
            "transcription": {"model": model_size}
        },
        "segments": results.get("segments", []),
        "text": results.get("text", ""),
        "language": results.get("language") or language_hint or "und",
        "timings_ms": timings,
        "diarization": diarization_result
    }
    return response
