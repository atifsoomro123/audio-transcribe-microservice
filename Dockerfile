# Dockerfile
FROM python:3.10-slim

# system deps: ffmpeg, build tools, libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
# install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy source
COPY src /app/src
COPY README.md /app/README.md
COPY sample_audio /app/sample_audio

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
