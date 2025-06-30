import torch
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, AutoTokenizer
from joblib import load
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# --- إعداد موديل speech-to-text ---
model_path = r"C:\Users\slman\Desktop\wav2vec2-local"
processor_wav2vec = Wav2Vec2Processor.from_pretrained(model_path)
model_wav2vec = Wav2Vec2ForCTC.from_pretrained(model_path)
model_wav2vec.eval()

# --- إعداد موديل تصحيح النص ---
t5_model_name = "willwade/t5-small-spoken-typo"
tokenizer_t5 = AutoTokenizer.from_pretrained(t5_model_name)
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
model_t5.eval()

# --- إعداد موديل تصنيف النية ---
model_intent = load(r'C:\Users\slman\Desktop\smart_car_backend\smart_assistance\models\car_intent_classifier.joblib')
lemmatizer = WordNetLemmatizer()

# --- دوال مساعدة ---

def correct_spelling_textblob(text):
    blob = TextBlob(text)
    return str(blob.correct())

def preprocess_text(text):
    # نصيحة: نص مُصحح من t5 ، لكن لنترك TextBlob كخطوة إضافية
    text = correct_spelling_textblob(text)  
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # إزالة علامات الترقيم
    tokens = word_tokenize(text, preserve_line=True)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def predict_intent(text):
    processed_text = preprocess_text(text)
    intent = model_intent.predict([processed_text])[0]
    probabilities = model_intent.predict_proba([processed_text])[0]
    return intent, probabilities

# --- تسجيل الصوت وتحويله إلى نص ---
def record_and_transcribe(duration=5, sample_rate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    waveform = torch.from_numpy(recording.squeeze())
    inputs = processor_wav2vec(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model_wav2vec(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_wav2vec.decode(predicted_ids[0])
    return transcription

# --- تصحيح النص باستخدام T5 ---
def correct_text_t5(input_text):
    inputs_t5 = tokenizer_t5(input_text, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model_t5.generate(**inputs_t5, max_length=128)
    processed_text = tokenizer_t5.decode(generated_ids[0], skip_special_tokens=True)
    return processed_text

# --- التشغيل المتسلسل ---
if __name__ == "__main__":
    # 1. تسجيل وتحويل الصوت إلى نص
    raw_text = record_and_transcribe(duration=5)
    print("Raw transcription:", raw_text)

    # 2. تصحيح النص بواسطة T5
    corrected_text = correct_text_t5(raw_text)
    print("Corrected text by T5:", corrected_text)

    # 3. توقع النية
    intent, probabilities = predict_intent(corrected_text)
    print(f"Intent prediction probabilities: {probabilities}")

    max_prob = max(probabilities)
    if max_prob < 0.5:
        print("The model is not sure about the intent (confidence < 50%)")
    else:
        print(f"Predicted intent: {intent}")
