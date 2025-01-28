from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TextClassificationPipeline

# Load pre-trained model
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt')

# Sample texts
texts = [
    "I used to have so many friends.",
    "But, With time, they got away from me.",
    "I really miss them. I remember the days of fun with them.",
    "But, Seems like they are off the road. They really don't care about me.",
    "I wanted to have a really good relatioship with them but, they are so far from me. Maybe it's better just not to go back to those moments"
]

results = pipeline(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
    print()