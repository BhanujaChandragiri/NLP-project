from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import TextClassificationPipeline

model_name = 'unitary/unbiased-toxic-roberta'
model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', return_all_scores=True)

# Sample texts
texts = [
    "All muslims has the tendecy to commit crime.",
    "Your family tree is a circle.",
    "The blacks should be wiped out from this world.",
    "You are a weird person!"
]

# Detect hate speech
results = pipeline(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    for label in result:
        print(f" - {label['label']}: {label['score']:.4f}")
    print()