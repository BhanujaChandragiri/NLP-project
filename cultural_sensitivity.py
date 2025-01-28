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

# Example usage
recognizer = CultureRecognizer()
text1 = "I love sushi and samurai movies."
print("Text 1:", recognizer.recognize_culture(text1))

text2 = "I'm wearing a beautiful sari for Diwali."
print("Text 2:", recognizer.recognize_culture(text2))

text3 = "Let's celebrate the Dragon Boat Festival."
print("Text 3:", recognizer.recognize_culture(text3))

text4 = "Bollywood is so big."
print("Text 4:", recognizer.recognize_culture(text4))