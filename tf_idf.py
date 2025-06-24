# tf_idf.py
from collections import Counter

def process_tfidf(user_input, tfidf_vectorizer_model, nlp_model_for_tokenization):
    """
    Calculates Term Frequency (TF) and TF-IDF scores for the input text.
    Args:
        user_input (str): The text to analyze.
        tfidf_vectorizer_model: The loaded TfidfVectorizer.
        nlp_model_for_tokenization: The loaded spaCy language model for tokenization.
    Returns:
        tuple: (tf_results: dict, relevant_tfidf_scores: dict)
    """
    tf_results = {}
    relevant_tfidf_scores = {}

    if tfidf_vectorizer_model is None or nlp_model_for_tokenization is None:
        print("TF-IDF vectorizer or spaCy model not loaded in tf_idf.py. Skipping TF-IDF analysis.")
        return {"error": "TF-IDF component not initialized."}, {"error": "TF-IDF component not initialized."}

    try:
        # Calculate Term Frequency (TF) for the input text
        # Ensure nlp_model_for_tokenization is used for tokenization
        words = [word.text.lower() for word in nlp_model_for_tokenization(user_input) if word.is_alpha and not word.is_stop]
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        if total_words > 0:
            tf_results = {word: count / total_words for word, count in word_counts.items()}
        
        # Transform the input text using the loaded TF-IDF vectorizer
        text_vectorized = tfidf_vectorizer_model.transform([user_input])
        
        # Get feature names (words) from the vectorizer
        feature_names = tfidf_vectorizer_model.get_feature_names_out()
        
        # Extract TF-IDF scores for words present in the input
        # Only iterate through non-zero entries for efficiency
        for col, score in zip(text_vectorized.indices, text_vectorized.data):
            word = feature_names[col]
            relevant_tfidf_scores[word] = score

        # Sort TF-IDF scores by score in descending order
        relevant_tfidf_scores = dict(sorted(relevant_tfidf_scores.items(), key=lambda item: item[1], reverse=True))

    except Exception as e:
        print(f"Error during TF-IDF calculation in tf_idf.py: {e}")
        tf_results = {"error": f"TF calculation failed: {e}"}
        relevant_tfidf_scores = {"error": f"TF-IDF calculation failed: {e}"}
    
    return tf_results, relevant_tfidf_scores

