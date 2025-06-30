import joblib
import re
from joblib import load
import subprocess
import sys
# For TTS
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play

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

def play_tts(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)

if __name__ == "__main__":
    test_text = "call mohamd"
    
    if test_text:
        intent_num, probabilities = predict_text(test_text)
        intent_name = intent_map.get(intent_num, f"Unknown intent ({intent_num})")

        print(f"Original text: {test_text}")
        print(f"Predicted intent (class): {intent_num}")
        print(f"Predicted intent (name): {intent_name}")

        if intent_num == 8:
            # Run the phone_call.py logic here and skip TTS
            print("Intent 8 detected: Forwarding to phone_call module.")
            subprocess.run([sys.executable, "smart_assistance/phone_call.py"])
        else:
            if probabilities is None:
                print("⚠️ Probabilities not available for this model.")
                print(f"Predicted intent: {intent_name}")
                play_tts(f"The detected intent is: {intent_name}")
            else:
                max_confidence = max(probabilities)
                if max_confidence < 0.6:
                    print(f"⚠️ Low confidence: {max_confidence:.2f} (< 60%)")
                    print("The model is not confident about the intent.")
                else:
                    print(f"Confidence: {max_confidence:.2f}")
                    print(f"All probabilities: {probabilities}")
                    play_tts(f"The detected intent is: {intent_name}")
