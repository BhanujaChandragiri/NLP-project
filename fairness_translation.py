from transformers import MarianMTModel, MarianTokenizer

# Load the MarianMT model English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Define a text to translated
english_text = "He is a doctor."

# Translate the English text to Hindi
translated_text = model.generate(**tokenizer(english_text, return_tensors="pt", padding=True))

# Decode the translated text
hindi_translation = tokenizer.decode(translated_text[0], skip_special_tokens=True)
print("English Text:", english_text)
print("Hindi Translation:", hindi_translation)