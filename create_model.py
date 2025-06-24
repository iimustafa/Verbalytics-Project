import spacy
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Define paths for saving models
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. TF-IDF Vectorizer training with cleaned_football_news.csv ---
print("Creating TF-IDF vectorizer using cleaned_football_news.csv...")
CSV_FILE_PATH = 'cleaned_football_news.csv'
TEXT_COLUMN_NAME = 'refined_content' # Confirmed column name

corpus_for_tfidf = []
if not os.path.exists(CSV_FILE_PATH):
    print(f"Error: {CSV_FILE_PATH} not found. Please ensure the CSV file is in the same directory.")
    print("Falling back to dummy corpus for TF-IDF (and sentiment text).")
    corpus_for_tfidf = [
        "football is a popular sport",
        "soccer is another name for football",
        "the match was exciting and the team played well",
        "sports news often covers football and basketball",
        "the team won the championship this year",
        "tennis is a great sport too"
    ]
else:
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        if TEXT_COLUMN_NAME not in df.columns:
            print(f"Error: Column '{TEXT_COLUMN_NAME}' not found in {CSV_FILE_PATH}.")
            print("Please update TEXT_COLUMN_NAME in create_model.py to the correct column name in your CSV.")
            print("Falling back to dummy corpus for TF-IDF (and sentiment text).")
            corpus_for_tfidf = [
                "football is a popular sport",
                "soccer is another name for football",
                "the match was exciting and the team played well",
                "sports news often covers football and basketball",
                "the team won the championship this year",
                "tennis is a great sport too"
            ]
        else:
            corpus_for_tfidf = df[TEXT_COLUMN_NAME].dropna().astype(str).tolist()
            print(f"Successfully loaded {len(corpus_for_tfidf)} documents from {CSV_FILE_PATH} for TF-IDF training.")
    except Exception as e:
        print(f"Error reading {CSV_FILE_PATH}: {e}")
        print("Falling back to dummy corpus for TF-IDF (and sentiment text).")
        corpus_for_tfidf = [
            "football is a popular sport",
            "soccer is another name for football",
            "the match was exciting and the team played well",
            "sports news often covers football and basketball",
            "the team won the championship this year",
            "tennis is a great sport too"
        ]

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_vectorizer.fit(corpus_for_tfidf)

with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"TF-IDF vectorizer saved to {os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')}")

# --- 2. Sentiment Analysis Model (LSTM) using text from CSV with random labels ---
print("\nCreating sentiment model and tokenizer with increased complexity...")
print("NOTE: Sentiment model will be trained on text from your CSV with *randomly generated labels*.")
print("      For accurate sentiment analysis, a dataset with true sentiment labels is required.")

# Use the corpus loaded from CSV for tokenizer fitting and model training
texts_for_sentiment_training = corpus_for_tfidf

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(texts_for_sentiment_training)

# Save tokenizer
with open(os.path.join(MODELS_DIR, 'sentiment_tokenizer.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
print(f"Sentiment tokenizer saved to {os.path.join(MODELS_DIR, 'sentiment_tokenizer.json')}")

# Create a more complex LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
max_len = 50 # This will be the fixed input length for the model

model = Sequential([
    Embedding(vocab_size, embedding_dim), # Removed input_length as it's deprecated
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    Dense(3, activation='softmax') # 3 classes: Positive, Negative, Neutral
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create padded sequences from your CSV text data
sequences = tokenizer.texts_to_sequences(texts_for_sentiment_training)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Generate random dummy labels for the training data (0, 1, or 2 for Negative, Neutral, Positive)
# The length of labels must match the number of samples
random_labels = np.random.randint(0, 3, size=len(texts_for_sentiment_training))

# Train the model with the actual text from your CSV and random labels
print("\nTraining sentiment model on CSV text with *random labels* (for structural validation, not performance)...")
model.fit(padded_sequences, random_labels, epochs=5, batch_size=32, verbose=0)

print("\nSentiment Model Architecture:")
model.summary() # Moved model.summary() to after model.fit()

# Save the model in the native Keras format
SENTIMENT_MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'sentiment_model.keras')
model.save(SENTIMENT_MODEL_SAVE_PATH)
print(f"\nSentiment model saved to {SENTIMENT_MODEL_SAVE_PATH}")

print("\nAll models created successfully in the 'models' directory.")
print("Remember to download the spaCy model 'en_core_web_sm' using: python -m spacy download en_core_web_sm")
print("\n--- IMPORTANT ---")
print("The sentiment model was trained with randomly generated labels on your text data.")
print("For meaningful sentiment analysis, you MUST provide a dataset with actual human-labeled sentiments.")
