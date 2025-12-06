import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------- 1. Prepare your corpus (list of documents) ----------
documents = [
    "I love this new phone, the camera quality is amazing",
    "The battery life of this phone is terrible and disappointing",
    "We had a great trip to the mountains and enjoyed the beautiful views",
    "The hotel room was dirty and the service was awful",
    "The food at the restaurant was delicious and the staff were friendly",
    "The movie was boring and too long, I almost fell asleep",
    "This new laptop is fast and perfect for my work",
    "Customer service was very helpful and resolved my issue quickly",
    "The flight was delayed and the seats were uncomfortable",
    "I had a wonderful time with friends at the beach",
]

# ---------- 2. Convert text → Document-Term Matrix ----------
# You can tweak min_df, max_df, stop_words, ngram_range, etc.
vectorizer = CountVectorizer(
    stop_words='english',  # remove common English words
    max_df=0.95,           # ignore very frequent words
    min_df=1               # keep words that appear in at least 1 document
)

dtm = vectorizer.fit_transform(documents)  # DTM = rows: docs, cols: words

# ---------- 3. Fit LDA model ----------
num_topics = 3  # choose how many topics you want
lda = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42
)
lda.fit(dtm)

# ---------- 4. Show top words per topic ----------
def print_topics(model, feature_names, num_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx}")
        top_indices = topic.argsort()[:-num_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        print("Top words:", ", ".join(top_words))

feature_names = vectorizer.get_feature_names_out()
print_topics(lda, feature_names)

# ---------- 5. Topic distribution for each document ----------
doc_topic_distribution = lda.transform(dtm)  # shape: (num_docs, num_topics)

print("\nDocument → topic probabilities:")
for i, probs in enumerate(doc_topic_distribution):
    print(f"Doc {i}: {probs}")
