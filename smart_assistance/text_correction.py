from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Better grammar correction model
model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

# Input with error
input_text = "trun off the headlights"

print("Input to grammar correction model:", input_text)

# Prefix needed for this model
input_ids = tokenizer.encode("gec: " + input_text, return_tensors="pt")

# Generate correction
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)

# Decode output
corrected_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Corrected text:", corrected_text)
