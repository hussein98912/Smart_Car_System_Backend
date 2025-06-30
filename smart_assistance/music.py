from rapidfuzz import process, fuzz
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play

# Function to play TTS
def play_tts(text):
    tts = gTTS(text=text, lang='en', slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)

# Your song dataset
songs = [
    "Bohemian Rhapsody",
    "Stairway to Heaven",
    "Hotel California",
    "Sweet Child O' Mine",
    "Smells Like Teen Spirit",
    "hard to stop",
    "emnim",
    "emnim for shakira"
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

    # First try matching the full text
    best_match, score, _ = process.extractOne(user_text, songs_lower, scorer=fuzz.token_sort_ratio)

    if score < 80:
        words = user_text.split()
        candidates = []
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words))+1):  # n-gram phrases
                phrase = " ".join(words[i:j])
                candidates.append(phrase)

        best_match = None
        best_score = 0

        for candidate in candidates:
            match, s, _ = process.extractOne(candidate, songs_lower, scorer=fuzz.token_sort_ratio)
            if s > best_score:
                best_score = s
                best_match = match
    else:
        best_score = score

    if best_score > 70:
        print(f"Detected song: {best_match}")
        play_tts(f"Playing {best_match}")
    else:
        print("No matching song found.")
        play_tts("Sorry, I couldn't find any matching song.")
else:
    print("No input text found in the file.")
    play_tts("No input text found.")
