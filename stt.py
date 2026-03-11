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

# ── Configuration ──────────────────────────────────────────────
VOSK_RATE = 16000
WAKE_WORD = "mira"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://mira-ollama:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "mira")
MODEL_PATH = os.getenv("VOSK_MODEL", "/app/model")
NOISE_THRESHOLD = int(os.getenv("NOISE_THRESHOLD", "800"))  # Seuil RMS, ajustable

# Variables MQTT
derniere_vision = "Rien à signaler"
last_vision_time = 0.0
mqtt_client = None

MOTOR_COMMANDS = {
    "avance", "avancer",
    "recule", "reculer", "recul",
    "autopilot", "autopilote",
    "stop", "stoppe", "arrête", "arreter",
    "gauche",
    "droite",
    "position",
}

# ── Couleurs terminal ─────────────────────────────────────────
C_RESET  = "\033[0m"
C_GREEN  = "\033[1;32m"
C_CYAN   = "\033[0;36m"
C_YELLOW = "\033[1;33m"
C_RED    = "\033[1;31m"
C_BLUE   = "\033[1;34m"

# ── Callbacks MQTT ────────────────────────────────────────────
def on_mqtt_connect(client, userdata, flags, rc):
    print(f"{C_CYAN}[MQTT] Connecté avec le code {rc}. Abonnement à mira/vision/output...{C_RESET}")
    client.subscribe("mira/vision/output")

def on_mqtt_message(client, userdata, msg):
    global derniere_vision, last_vision_time
    derniere_vision = msg.payload.decode("utf-8")
    last_vision_time = time.time()

# ── Queue audio ───────────────────────────────────────────────
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Callback appelé par sounddevice pour chaque bloc audio."""
    if status:
        print(f"{C_YELLOW}[AUDIO] {status}{C_RESET}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def compute_rms(data):
    """Calcule le niveau RMS (volume) d'un chunk audio int16."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0
    return np.sqrt(np.mean(samples ** 2))

def noise_gate(data, threshold):
    """Remplace le chunk par du silence si le volume est sous le seuil."""
    rms = compute_rms(data)
    if rms < threshold:
        return b'\x00' * len(data), rms, True
    return data, rms, False

def downsample(data, from_rate, to_rate):
    """Downsampling simple (prend 1 échantillon sur N) de int16 PCM."""
    if from_rate == to_rate:
        return data
    ratio = from_rate / to_rate
    samples = struct.unpack(f"<{len(data)//2}h", data)
    new_samples = []
    pos = 0.0
    while int(pos) < len(samples):
        new_samples.append(samples[int(pos)])
        pos += ratio
    return struct.pack(f"<{len(new_samples)}h", *new_samples)

def detect_motor_command(text):
    """Vérifie si le texte contient un ordre moteur connu."""
    words = text.lower().split()
    for word in words:
        if word in MOTOR_COMMANDS:
            return word
    return None

def ask_ollama(prompt):
    """Envoie une requête au LLM Ollama et retourne la réponse."""
    try:
        print(f"{C_CYAN}[LLM] Envoi au modèle '{MODEL_NAME}'...{C_RESET}")
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=120)
        response = r.json().get("response", "Erreur: pas de réponse.")
        return response
    except requests.exceptions.ConnectionError:
        return "Erreur: impossible de joindre Ollama."
    except Exception as e:
        return f"Erreur LLM: {e}"

def process_text(text):
    """Traite le texte reconnu après détection du wake word."""
    text_lower = text.lower().strip()

    # Cherche "mira" dans le texte
    if WAKE_WORD not in text_lower:
        return  # Pas de wake word, on ignore

    # Extrait ce qui suit "mira"
    idx = text_lower.index(WAKE_WORD) + len(WAKE_WORD)
    after_wake = text_lower[idx:].strip()

    if not after_wake:
        print(f"{C_YELLOW}[WAKE] Wake word détecté mais pas de commande.{C_RESET}")
        return

    print(f"{C_GREEN}[WAKE] Commande détectée : \"{after_wake}\"{C_RESET}")

    # Vérifie si c'est un ordre moteur
    motor_cmd = detect_motor_command(after_wake)
    if motor_cmd:
        print(f"{C_RED}[ORDRE DÉTECTÉ] {motor_cmd.upper()}{C_RESET}")
        if mqtt_client:
            payload = json.dumps({"action": motor_cmd})
            mqtt_client.publish("mira/bridge/ordres", payload)
            print(f"{C_CYAN}[MQTT] Ordre publié : {payload}{C_RESET}")
        return

    # Sinon, c'est une question → envoyer au LLM dans un thread séparé
    # pour ne pas bloquer l'écoute micro
    print(f"{C_BLUE}[QUESTION] \"{after_wake}\"{C_RESET}")
    threading.Thread(target=_ask_and_print, args=(after_wake,), daemon=True).start()

def _ask_and_print(prompt):
    """Thread worker pour appeler Ollama sans bloquer la boucle audio."""
    global derniere_vision, last_vision_time
    
    vision_text = "Rien à signaler"
    if time.time() - last_vision_time <= 15:
        vision_text = derniere_vision
        
    full_prompt = (
        f"[VUE (Caméra)] : {vision_text}\n"
        f"[AUDIO (Microphone)] : {prompt}\n"
        f"Réponds de manière naturelle et concise en tant que M.I.R.A."
    )
    
    response = ask_ollama(full_prompt)
    print(f"{C_GREEN}[MIRA] {response}{C_RESET}")

def find_microphone():
    """Trouve le micro USB ou utilise le micro par défaut."""
    devices = sd.query_devices()
    usb_device = None
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']} (entrées: {dev['max_input_channels']})")
            if "usb" in dev["name"].lower():
                usb_device = i
    device_id = usb_device if usb_device is not None else sd.default.device[0]
    return device_id

def find_working_rate(device_id):
    """Essaie plusieurs sample rates pour trouver celui supporté par le micro."""
    rates_to_try = [16000, 44100, 48000, 22050, 8000, 32000, 96000]
    for rate in rates_to_try:
        try:
            sd.check_input_settings(device=device_id, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue
    return None

def main():
    global mqtt_client
    
    # 0. Initialisation MQTT
    print(f"{C_CYAN}[INIT] Connexion au broker MQTT...{C_RESET}")
    # Compatibilité paho-mqtt v1 et v2
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        mqtt_client = mqtt.Client()
        
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect("mira-mosquitto", 1883, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"{C_RED}[ERREUR] Impossible de se connecter à mira-mosquitto : {e}{C_RESET}")

    # 1. Charger le modèle Vosk
    print(f"{C_CYAN}[INIT] Chargement du modèle Vosk...{C_RESET}")
    if not os.path.exists(MODEL_PATH):
        print(f"{C_RED}[ERREUR] Modèle Vosk introuvable à {MODEL_PATH}{C_RESET}")
        sys.exit(1)

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, VOSK_RATE)

    # 2. Trouver le micro
    print(f"{C_CYAN}[INIT] Recherche du micro...{C_RESET}")
    device_id = find_microphone()
    dev_info = sd.query_devices(device_id)
    print(f"{C_GREEN}[INIT] Micro sélectionné : {dev_info['name']}{C_RESET}")

    # 3. Trouver un sample rate supporté
    device_rate = find_working_rate(device_id)
    if device_rate is None:
        print(f"{C_RED}[ERREUR] Aucun sample rate supporté trouvé pour ce micro.{C_RESET}")
        sys.exit(1)

    needs_resample = (device_rate != VOSK_RATE)
    print(f"{C_CYAN}[INIT] Sample rate micro: {device_rate} Hz" +
          (f" (resample → {VOSK_RATE} Hz)" if needs_resample else "") + f"{C_RESET}")

    # 4. Boucle d'écoute
    print(f"{C_GREEN}>>> PRÊT. J'écoute en continu (wake word: '{WAKE_WORD}')...{C_RESET}")

    with sd.RawInputStream(
        samplerate=device_rate,
        blocksize=16000,
        device=device_id,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()
            if needs_resample:
                data = downsample(data, device_rate, VOSK_RATE)
            # Noise gate : on envoie du silence si le son est trop faible
            data, rms, is_silent = noise_gate(data, NOISE_THRESHOLD)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"{C_CYAN}[STT] \"{text}\"{C_RESET}")
                    process_text(text)

if __name__ == "__main__":
    main()
