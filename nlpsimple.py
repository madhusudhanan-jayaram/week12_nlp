from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER if not already downloaded
nltk.download('vader_lexicon')

# Create analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)

    compound = scores['compound']

    if compound >= 0.05:
        sentiment = "POSITIVE"
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    print("\nText:", text)
    print("Scores:", scores)
    print("Sentiment:", sentiment)


# Test examples
sentences = [
    "I love this product! It's amazing 😍",
    "Terrible experience. I hate it.",
    "The movie was okay, nothing special.",
    "Not bad at all!",
    "I am sooo happy with the service!!!",
    "Worst thing ever 🤮",
]

for s in sentences:
    analyze_sentiment(s)


# Interactive mode
print("\n--- Type 'q' to quit ---")
while True:
    user_input = input("Enter a sentence: ")
    if user_input.lower() == "q":
        break
    analyze_sentiment(user_input)
