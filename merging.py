import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load BERT model and tokenizer for masked word prediction
model_name = 'bert-base-uncased'
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load MarianMT model and tokenizer for English to Hindi translation
translation_model_name = "Helsinki-NLP/opus-mt-en-hi"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Load Roberta model and tokenizer for hate speech detection
hate_speech_model_name = 'unitary/unbiased-toxic-roberta'
hate_speech_model = RobertaForSequenceClassification.from_pretrained(hate_speech_model_name)
hate_speech_tokenizer = RobertaTokenizer.from_pretrained(hate_speech_model_name)
hate_speech_pipeline = TextClassificationPipeline(model=hate_speech_model, tokenizer=hate_speech_tokenizer, framework='pt', return_all_scores=True)

# Load DistilBert model and tokenizer for sentiment analysis
sentiment_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_model = DistilBertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(sentiment_model_name)
sentiment_pipeline = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer, framework='pt')

# Function to predict masked words
def predict_masked_words(text, top_k=10):
    inputs = bert_tokenizer(text, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == bert_tokenizer.mask_token_id)[1]
    token_logits = bert_model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
    return [bert_tokenizer.decode([token]).strip() for token in top_k_tokens]

# Function to recognize culture from text
class CultureRecognizer:
    def __init__(self):
        self.culture_keywords = {
            'Japanese': ['sakura', 'kimono', 'sushi', 'samurai'],
            'Indian': ['holi', 'sari', 'bollywood', 'curry'],
            'Chinese': ['dragon', 'lantern', 'chopsticks', 'kung fu']
        }

    def recognize_culture(self, text):
        text = text.lower()
        for culture, keywords in self.culture_keywords.items():
            if any(keyword in text for keyword in keywords):
                return culture
        return "Culture not recognized."

# Function to translate text from English to Hindi
def translate_to_hindi(english_text):
    translated_text = translation_model.generate(**translation_tokenizer(english_text, return_tensors="pt", padding=True))
    hindi_translation = translation_tokenizer.decode(translated_text[0], skip_special_tokens=True)
    return hindi_translation

# Class to suggest inclusive language
class InclusiveLanguageModel:
    def __init__(self, inclusive_rules):
        self.inclusive_rules = inclusive_rules

    def suggest_inclusive_language(self, text):
        suggested_text = text
        for old, new in self.inclusive_rules.items():
            suggested_text = suggested_text.replace(old, new)
        return suggested_text

# Define inclusive rules
inclusive_rules = {
    "chairman": "chairperson",
    "policeman": "police officer",
    "fireman": "firefighter",
    "man-made": "artificial",
    "What the hell": "This is not good",
    "dead": "pass away",
    "weird": "not so bright"
}

# Main function to use the functionalities
def process_text(text):
    results = {}

    # Masked word prediction
    if '[MASK]' in text:
        predictions = predict_masked_words(text)
        results['masked_word_predictions'] = predictions

    # Culture recognition
    recognizer = CultureRecognizer()
    recognized_culture = recognizer.recognize_culture(text)
    results['recognized_culture'] = recognized_culture

    # English to Hindi translation
    hindi_translation = translate_to_hindi(text)
    results['hindi_translation'] = hindi_translation

    # Hate speech detection
    hate_speech_results = hate_speech_pipeline([text])[0]
    results['hate_speech_detection'] = {label['label']: label['score'] for label in hate_speech_results}

    # Inclusive language suggestion
    language_model = InclusiveLanguageModel(inclusive_rules)
    suggested_text = language_model.suggest_inclusive_language(text)
    results['inclusive_language_suggestion'] = suggested_text

    # Sentiment analysis
    sentiment_result = sentiment_pipeline([text])[0]
    results['sentiment_analysis'] = {
        'label': sentiment_result['label'],
        'score': sentiment_result['score']
    }

    return results

# Example usage
text = input("Enter your text here: ")
output = process_text(text)
print(output)
