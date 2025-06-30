# smart_assistance/wake_word_service.py
import requests
import os
import queue
import sounddevice as sd
import vosk
import json
import subprocess
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
import time

# === MODEL PATH ===
MODEL_PATH = r"smart_assistance/vosk-model-small-en-us-0.15"

# === CHECK IF MODEL EXISTS ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found at {}".format(MODEL_PATH))

# === LOAD THE VOSK MODEL ===
model = vosk.Model(MODEL_PATH)

# === AUDIO QUEUE ===
q = queue.Queue()

# === AUDIO CALLBACK FUNCTION ===
def callback(indata, frames, time, status):
    if status:
        print("Audio status:", status)
    q.put(bytes(indata))

# === PLAY TTS WITHOUT SAVING FILE ===
def play_tts(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)

# === MAIN FUNCTION ===
def start_listening():
    print("LLLLLLLLLLListening for wake word: 'hi google'...")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback): #stream for micro

        recognizer = vosk.KaldiRecognizer(model, 16000)

        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
                print("Recognized:", text)

                if "hi" in text and "google" in text:
                    print("Wake word detected.")
                    with open("wake_word_status.txt", "w") as f:
                        f.write("0")
                    play_tts("i catch a request from you")
                    time.sleep(30)  # تأخير مؤقت
                    with open("wake_word_status.txt", "w") as f:
                        f.write("1")
                    subprocess.run(["python", "smart_assistance/speech_to_text.py"])
                    break
