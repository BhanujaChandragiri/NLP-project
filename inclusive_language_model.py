class InclusiveLanguageModel:
    def __init__(self, inclusive_rules):
        self.inclusive_rules = inclusive_rules

    def suggest_inclusive_language(self, text):
        suggested_text = text
        for old, new in self.inclusive_rules.items():
            suggested_text = suggested_text.replace(old, new)
        return suggested_text

# Example usage
inclusive_rules = {
    "chairman": "chairperson",
    "policeman": "police officer",
    "fireman": "firefighter",
    "man-made": "artificial",
    "What the hell": "This is not good",
    "dead": "pass away",
    "weird": "not so bright"
}

language_model = InclusiveLanguageModel(inclusive_rules)

text = input("Give your text here: ")
suggested_text = language_model.suggest_inclusive_language(text)
print("Original Text:", text)
print("Suggested Text:", suggested_text)
