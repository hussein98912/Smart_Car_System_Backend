import joblib
import re
from joblib import load
import subprocess
import sys
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play
from rapidfuzz import process, fuzz
import numpy as np

# Load the model
model = load(r'smart_assistance\models\text_classifier_model_with_proba (1).joblib')

# === Mapping intent numbers to intent names ===
intent_map = {
    1: "Turn off headlights",
    2: "Turn on headlights",
    3: "Car door lock",
    4: "Car door open",
    5: "Car windows close",
    6: "Car windows open",
    7: "Music dislikeness",
    8: "Make phone call",
    14: "Audio volume up",
    35: "Audio volume down",
    36: "Play radio",
    43: "Music likeness",
    46: "Audio volume mute",
    57: "Music query"
}

# Song dataset
songs = [
    "Bohemian Rhapsody",
    "Stairway to Heaven",
    "Hotel California",
    "Sweet Child O' Mine",
    "Smells Like Teen Spirit",
    "hard to stop",
    "emnim",
    "emnim for shakira"
    # Add more songs...
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text):
    if isinstance(text, str):
        text = [text]
    cleaned_texts = [clean_text(t) for t in text]
    try:
        probabilities = model.predict_proba(cleaned_texts)[0]
        predicted_class = model.predict(cleaned_texts)[0]
        return predicted_class, probabilities
    except AttributeError:
        predicted_class = model.predict(cleaned_texts)[0]
        return predicted_class, None

def detect_song(text, songs, threshold=70):
    text = text.lower()
    songs_lower = [s.lower() for s in songs]

    best_match, score, _ = process.extractOne(text, songs_lower, scorer=fuzz.token_sort_ratio)

    if score < 80:
        words = text.split()
        candidates = []
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                candidates.append(phrase)

        best_score = 0
        best_match = None
        for candidate in candidates:
            match, score, _ = process.extractOne(candidate, songs_lower, scorer=fuzz.token_sort_ratio)
            if score > best_score:
                best_score = score
                best_match = match

    if best_score > threshold:
        return best_match
    else:
        return None

def play_tts(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)

if __name__ == "__main__":
    input_path = r"smart_assistance\live_input.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        test_text = f.read().strip()

    if test_text:
        # --- Song detection logic ---
        detected_song = detect_song(test_text, songs)
        if detected_song:
            print(f"🎵 Detected song in input: {detected_song}")
            intent_num = 57
            probabilities = [0.0] * len(intent_map)
            max_confidence = 0.80  # Set confidence artificially
        else:
            # Regular intent classification
            intent_num, probabilities = predict_text(test_text)
            if probabilities is not None and len(probabilities) > 0:
                max_confidence = float(np.max(probabilities))
            else:
                max_confidence = None

        intent_name = intent_map.get(intent_num, f"Unknown intent ({intent_num})")
        print(f"Original text: {test_text}")
        print(f"Predicted intent (class): {intent_num}")
        print(f"Predicted intent (name): {intent_name}")

        if probabilities is None:
            print("⚠️ Probabilities not available for this model.")
            print(f"Predicted intent: {intent_name}")
            play_tts(f"The detected intent is: {intent_name}")
        else:
            print(f"Confidence: {max_confidence:.2f}")
            print(f"All probabilities: {probabilities}")

            # Check conditions
            if (intent_num == 57 and max_confidence >= 0.4) or (intent_num != 57 and max_confidence >= 0.6):
                if intent_num == 8:
                    print("Intent 8 detected: Forwarding to phone_call module.")
                    subprocess.run([sys.executable, "smart_assistance/phone_call.py"])
                elif intent_num == 57:
                    print("Intent 57 detected: Forwarding to music module.")
                    subprocess.run([sys.executable, "smart_assistance/music.py"])
                else:
                    play_tts(f"The detected intent is: {intent_name}")
            else:
                print(f"⚠️ Confidence too low for intent {intent_num}: {max_confidence:.2f}")
                print("The model is not confident enough to act.")
                play_tts("Sorry, I didn't understand you.")
def predict_intent_from_file():
    input_path = r"smart_assistance\live_input.txt"
    with open(input_path, "r", encoding="utf-8") as f:
        test_text = f.read().strip()

    if not test_text:
        return None, None, None

    detected_song = detect_song(test_text, songs)
    if detected_song:
        intent_num = 57
        intent_name = intent_map.get(intent_num, "Music query")
        max_confidence = 0.80
    else:
        intent_num, probabilities = predict_text(test_text)
        intent_name = intent_map.get(intent_num, f"Unknown intent ({intent_num})")
        max_confidence = float(np.max(probabilities)) if probabilities is not None else None

    return intent_num, intent_name, max_confidence