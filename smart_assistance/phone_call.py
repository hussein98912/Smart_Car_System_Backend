from rapidfuzz import process, fuzz
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play

def play_tts(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)

# Your songs list here
songs = [
    "mohamd almouazen",
    "sami alhusisni",
    "hussin salman",
    "home",
    "mama",
    "dad",
    # ... 95 more songs
]

# Read user input from file
input_file = "smart_assistance/live_input.txt"
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        user_text = f.read().strip()
except FileNotFoundError:
    print(f"Input file '{input_file}' not found.")
    user_text = ""

if user_text:
    user_text = user_text.lower()
    songs_lower = [s.lower() for s in songs]

    best_match, score, _ = process.extractOne(user_text, songs_lower, scorer=fuzz.token_sort_ratio)

    if score < 80:
        words = user_text.split()
        candidates = []
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):
                phrase = " ".join(words[i:j])
                candidates.append(phrase)

        best_match = None
        best_score = 0

        for candidate in candidates:
            match, score, _ = process.extractOne(candidate, songs_lower, scorer=fuzz.token_sort_ratio)
            if score > best_score:
                best_score = score
                best_match = match
    else:
        best_score = score

    if best_score > 70:
        print(f"Detected the phone number name: {best_match}")
        # Play TTS message
        play_tts(f"I will call to {best_match}")
    else:
        print("No matching phone number found.")
else:
    print("No input text found in the file.")
