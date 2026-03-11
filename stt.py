import os
import sys
import json
import queue
import struct
import threading
import time
import numpy as np
import requests
import sounddevice as sd
import paho.mqtt.client as mqtt
from vosk import Model, KaldiRecognizer

VOSK_RATE = 16000
WAKE_WORDS = ["mira", "miro"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://100.68.211.25:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mira")
MODEL_PATH = os.getenv("VOSK_MODEL", "/app/model")
NOISE_THRESHOLD = int(os.getenv("NOISE_THRESHOLD", "300"))

derniere_vision = "Rien à signaler"
last_vision_time = 0.0
mqtt_client = None

MOTOR_COMMANDS = {
    "avance", "avancer", "recule", "reculer", "recul",
    "autopilot", "autopilote", "stop", "stoppe", "arrête", 
    "arreter", "gauche", "droite", "position",
}

C_RESET, C_GREEN, C_CYAN, C_YELLOW, C_RED, C_BLUE = (
    "\033[0m", "\033[1;32m", "\033[0;36m", "\033[1;33m", "\033[1;31m", "\033[1;34m"
)

def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    print(f"{C_CYAN}[MQTT] Connecté code {rc}.{C_RESET}")
    client.subscribe("mira/vision/output")

def on_mqtt_message(client, userdata, msg):
    global derniere_vision, last_vision_time
    derniere_vision = msg.payload.decode("utf-8")
    last_vision_time = time.time()

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status: print(f"{C_YELLOW}[AUDIO] {status}{C_RESET}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def compute_rms(data):
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(samples ** 2)) if len(samples) > 0 else 0

def noise_gate(data, threshold):
    rms = compute_rms(data)
    if rms < threshold: return b'\x00' * len(data), rms, True
    return data, rms, False

def downsample(data, from_rate, to_rate):
    if from_rate == to_rate: return data
    samples = np.frombuffer(data, dtype=np.int16)
    ratio = from_rate // to_rate
    return samples[::ratio].tobytes()

def detect_motor_command(text):
    words = text.lower().split()
    for word in words:
        if word in MOTOR_COMMANDS: return word
    return None

def ask_ollama(prompt):
    try:
        print(f"{C_CYAN}[LLM] Envoi...{C_RESET}")
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=120)
        return r.json().get("response", "...")
    except Exception as e: return f"Erreur: {e}"

def process_text(text):
    text_lower = text.lower().strip()
    
    found_wake = next((w for w in WAKE_WORDS if w in text_lower), None)
    if not found_wake: return

    idx = text_lower.index(found_wake) + len(found_wake)
    after_wake = text_lower[idx:].strip()

    if not after_wake:
        print(f"{C_YELLOW}[WAKE] {found_wake} détecté.{C_RESET}")
        return

    print(f"{C_GREEN}[WAKE] Commande : \"{after_wake}\"{C_RESET}")

    motor_cmd = detect_motor_command(after_wake)
    if motor_cmd:
        if mqtt_client:
            mqtt_client.publish("mira/bridge/ordres", json.dumps({"action": motor_cmd}))
        return

    threading.Thread(target=_ask_and_print, args=(after_wake,), daemon=True).start()

def _ask_and_print(prompt):
    global derniere_vision, last_vision_time
    ctx = derniere_vision if (time.time() - last_vision_time <= 15) else "Rien à signaler"
    full_prompt = f"[VUE]: {ctx}\n[AUDIO]: {prompt}\nRéponds brièvement."
    
    response = ask_ollama(full_prompt)
    
    print(f"{C_GREEN}[MIRA] {response}{C_RESET}")
    
    if mqtt_client:
        mqtt_client.publish("mira/stt/reponse", response)

def main():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        mqtt_client.connect("mira-mosquitto", 1883, 60)
        mqtt_client.loop_start()
    except: pass

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, VOSK_RATE)
    device_id = sd.default.device[0]
    device_rate = int(sd.query_devices(device_id)['default_samplerate'])

    print(f"{C_GREEN}>>> M.I.R.A PRÊTE ({WAKE_WORDS}){C_RESET}")

    with sd.RawInputStream(samplerate=device_rate, blocksize=32000, device=device_id,
                           dtype="int16", channels=1, callback=audio_callback):
        while True:
            data = audio_queue.get()
            data = downsample(data, device_rate, VOSK_RATE)
            data, _, _ = noise_gate(data, NOISE_THRESHOLD)
            if recognizer.AcceptWaveform(data):
                res = json.loads(recognizer.Result())
                text = res.get("text", "")
                if text:
                    print(f"{C_CYAN}[STT] \"{text}\"{C_RESET}")
                    process_text(text)

if __name__ == "__main__":
    main()