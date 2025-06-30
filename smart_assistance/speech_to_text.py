import os
import logging
import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import pipeline
import subprocess
import sys

from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play


# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# === Step 1: Record Audio from Microphone ===
SAMPLE_RATE = 16000  # Whisper accepts 16kHz input
DURATION = 10  # seconds

# Play a welcome message before recording
welcome_text = "Hello, how can I assist you?"
tts = gTTS(text=welcome_text, lang='en', slow=False)
mp3_fp = io.BytesIO()
tts.write_to_fp(mp3_fp)
mp3_fp.seek(0)
audio = AudioSegment.from_file(mp3_fp, format="mp3")
play(audio)

print("Recording... Speak now.")
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("Recording finished.")

# Save the recording as WAV
recorded_file = "smart_assistance/live_input.wav"
os.makedirs(os.path.dirname(recorded_file), exist_ok=True)
wav.write(recorded_file, SAMPLE_RATE, recording)

# === Step 2: Load Whisper ASR Model and Transcribe ===
print("Transcribing audio...")
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=0,  # use device=-1 for CPU, 0 for GPU (adjust as needed)
    generate_kwargs={"language": "en"}  # force English
)

result = asr(recorded_file)
transcribed_text = result.get("text", "").strip()
print("Transcribed Text:", transcribed_text)

# === Step 3: Save transcription to text file ===
text_output_file = "smart_assistance/live_input.txt"
with open(text_output_file, "w", encoding="utf-8") as f:
    f.write(transcribed_text)
print(f"Transcription saved to {text_output_file}")

# === Step 4: Run the next script ===
next_script = os.path.abspath("smart_assistance/text_to_command.py")
print(f" Running next script: {next_script}")

try:
    subprocess.run(["python", next_script], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running {next_script}: {e}")
    sys.exit(1)
