# sentiment.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def analyze_sentiment(user_input, sentiment_model, sentiment_tokenizer, max_len):
    """
    Performs sentiment analysis on the input text using a loaded Keras model.
    Args:
        user_input (str): The text to analyze.
        sentiment_model: The loaded Keras sentiment model.
        sentiment_tokenizer: The loaded Keras Tokenizer.
        max_len (int): The maximum sequence length for padding, consistent with model training.
    Returns:
        tuple: (sentiment_label: str, sentiment_polarity: float or str)
               sentiment_label can be "Positive", "Negative", "Neutral", or "Model Error".
               sentiment_polarity is the calculated score or "N/A".
    """
    if sentiment_model is None or sentiment_tokenizer is None:
        print("Sentiment model or tokenizer not loaded. Cannot perform sentiment analysis.")
        return "Model Not Loaded / Error", "N/A"

    try:
        # Preprocess the input text for the sentiment model
        seq = sentiment_tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        # Predict sentiment probabilities
        prediction = sentiment_model.predict(padded_seq)
        # Assuming 3 classes (Negative, Neutral, Positive) in the output softmax layer
        # Polarity calculation based on the difference between positive and negative probabilities
        polarity_score = float(prediction[0][2] - prediction[0][0]) 

        # Determine sentiment label based on highest probability
        predicted_class = np.argmax(prediction[0])

        if predicted_class == 2: # Assuming 2 for Positive
            sentiment_label = "Positive"
        elif predicted_class == 0: # Assuming 0 for Negative
            sentiment_label = "Negative"
        else: # Assuming 1 for Neutral
            sentiment_label = "Neutral"

        return sentiment_label, polarity_score

    except Exception as e:
        print(f"Error during sentiment analysis in sentiment.py: {e}")
        return "Model Not Loaded / Error", "N/A"

