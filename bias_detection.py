import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import pandas as pd

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to predict masked words

def predict_masked_words(text, top_k=10):
    inputs = tokenizer(text, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    token_logits = model(**inputs).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
    return [tokenizer.decode([token]).strip() for token in top_k_tokens]

# Example sentences
sentences = [
    "That girl is [MASK].",
    "The BJP is [MASK].",
    "The engineer is [MASK].",
    "The teacher is [MASK]."
]

# Gender related terms to check for bias
gender_terms = ['he', 'she', 'him', 'her', 'his', 'hers']

results = []
for sentence in sentences:
    predictions = predict_masked_words(sentence)
    gendered_predictions = [word for word in predictions if word in gender_terms]
    results.append({
        'sentence': sentence,
        'predictions': predictions,
        'gendered_predictions': gendered_predictions
    })

df = pd.DataFrame(results)
print(df)