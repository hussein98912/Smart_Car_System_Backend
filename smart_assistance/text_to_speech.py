from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

# Text you want to convert to speech
text = "Hello, how can i assist you today"

# Create gTTS object
tts = gTTS(text=text, lang='en', slow=False)

# Save to in-memory bytes buffer
mp3_fp = io.BytesIO()
tts.write_to_fp(mp3_fp)
mp3_fp.seek(0)

# Load audio from bytes buffer using pydub
audio = AudioSegment.from_file(mp3_fp, format="mp3")

# Play the audio directly
play(audio)
