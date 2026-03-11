FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    alsa-utils \
    libatomic1 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Télécharger le modèle Vosk français (full, ~1.4 GB, bien plus précis)
RUN wget -q https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip && \
    unzip vosk-model-fr-0.22.zip && \
    mv vosk-model-fr-0.22 model && \
    rm vosk-model-fr-0.22.zip

COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY stt.py .

ENV VOSK_MODEL=/app/model
ENV OLLAMA_URL=http://mira-ollama:11434/api/generate
ENV OLLAMA_MODEL=mira

CMD ["python", "-u", "stt.py"]
