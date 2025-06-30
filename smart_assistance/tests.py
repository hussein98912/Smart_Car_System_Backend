import torch
import torchaudio
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the processor and model
model_path = r"C:\Users\slman\Desktop\wav2vec2-local"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

# Define the recording parameters
duration = 5  # seconds
sample_rate = 16000  # Hz

print("Recording...")
# Record audio from the microphone
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  # Wait until recording is finished
print("Recording complete.")

# Convert the NumPy array to a PyTorch tensor
waveform = torch.from_numpy(recording.squeeze())

# Tokenize the audio
inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Decode the predicted IDs to text
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)
