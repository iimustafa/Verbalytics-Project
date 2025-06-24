from flask import Flask, render_template, request, make_response # Added make_response
import pickle
import json
import os
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Import functions from NLP modules
# Ensure these function names match what's defined in their respective files
from name_entity import perform_ner
from pos import perform_pos_tagging
from sentiment import analyze_sentiment
from tf_idf import process_tfidf

app = Flask(__name__)

# Global variables for NLP models
nlp = None # SpaCy model for NER and POS
tfidf_vectorizer = None # TF-IDF model
sentiment_model = None # Keras sentiment model
sentiment_tokenizer = None # Keras sentiment tokenizer
sentiment_max_len = 50 # IMPORTANT: Matches create_model.py for sentiment input padding

# Define paths to models - ensure these match what create_model.py saves
MODELS_DIR = 'models'
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.keras') # Expecting .keras now
SENTIMENT_TOKENIZER_PATH = os.path.join(MODELS_DIR, 'sentiment_tokenizer.json')

def load_nlp_models():
    """Loads all NLP models at application startup."""
    global nlp, tfidf_vectorizer, sentiment_model, sentiment_tokenizer

    # Load spaCy model for NER and POS tagging
    try:
        print("Loading spaCy model 'en_core_web_sm'...")
        nlp = spacy.load("en_core_web_sm")
        print("SpaCy model loaded successfully.")
    except Exception as e:
        print(f"Error loading spaCy model: {e}. Please ensure it's downloaded (python -m spacy download en_core_web_sm).")
        nlp = None

    # Load TF-IDF vectorizer
    try:
        print("Loading TF-IDF vectorizer...")
        with open(TFIDF_VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print("TF-IDF vectorizer loaded successfully.")
    except FileNotFoundError:
        print(f"TF-IDF vectorizer not found at {TFIDF_VECTORIZER_PATH}. Please run create_model.py.")
        tfidf_vectorizer = None
    except Exception as e:
        print(f"Error loading TF-IDF vectorizer: {e}")
        tfidf_vectorizer = None

    # Load Sentiment Analysis model and tokenizer
    try:
        print("Loading sentiment analysis model and tokenizer...")
        sentiment_model = load_model(SENTIMENT_MODEL_PATH)
        with open(SENTIMENT_TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
            sentiment_tokenizer = tokenizer_from_json(tokenizer_json)
        print("Sentiment analysis model and tokenizer loaded successfully.")
    except FileNotFoundError:
        print(f"Sentiment model or tokenizer not found. Ensure {SENTIMENT_MODEL_PATH} and {SENTIMENT_TOKENIZER_PATH} exist. Please run create_model.py.")
        sentiment_model = None
        sentiment_tokenizer = None
    except Exception as e:
        print(f"Error loading sentiment model/tokenizer: {e}")
        sentiment_model = None
        sentiment_tokenizer = None

# Load models when the Flask app starts within the application context
with app.app_context():
    load_nlp_models()


# --- Theme Management ---
def get_theme():
    """Gets the theme preference from cookies, defaults to 'dark'."""
    return request.cookies.get('theme', 'dark')

@app.route('/set_theme', methods=['POST'])
def set_theme():
    """Sets the theme preference cookie."""
    data = request.get_json()
    theme = data.get('theme', 'dark')
    response = make_response("Theme updated")
    response.set_cookie('theme', theme, max_age=30*24*60*60) # Cookie lasts 30 days
    return response


# --- Routes for Welcome Page and Individual Function Pages (GET requests) ---

@app.route('/')
def index():
    """Renders the main welcome/NLP Hub page."""
    current_theme = get_theme()
    return render_template('index.html', theme=current_theme)

@app.route('/ner')
def ner_page():
    """Renders the Named Entity Recognition page."""
    current_theme = get_theme()
    return render_template('ner.html', theme=current_theme)

@app.route('/pos')
def pos_page():
    """Renders the Part-of-Speech Tagging page."""
    current_theme = get_theme()
    return render_template('pos.html', theme=current_theme)

@app.route('/tfidf')
def tfidf_page():
    """Renders the TF-IDF Analysis page."""
    current_theme = get_theme()
    return render_template('tfidf.html', theme=current_theme)

@app.route('/sentiment')
def sentiment_page():
    """Renders the Sentiment Analysis page."""
    current_theme = get_theme()
    return render_template('sentiment.html', theme=current_theme)


# --- Analysis Routes (POST requests) ---

@app.route('/analyze_ner', methods=['POST'])
def analyze_ner_route():
    user_input = request.form['text_input']
    entities = perform_ner(user_input, nlp) 
    current_theme = get_theme()
    return render_template(
        'ner.html',
        ner_input=user_input,
        ner_entities=entities,
        theme=current_theme
    )

@app.route('/analyze_pos', methods=['POST'])
def analyze_pos_route():
    user_input = request.form['text_input']
    pos_tags = perform_pos_tagging(user_input, nlp) 
    current_theme = get_theme()
    return render_template(
        'pos.html',
        pos_input=user_input,
        pos_tags=pos_tags,
        theme=current_theme
    )

@app.route('/analyze_tfidf', methods=['POST'])
def analyze_tfidf_route():
    user_input = request.form['text_input']
    tf_results, relevant_tfidf_scores = process_tfidf(user_input, tfidf_vectorizer, nlp)
    current_theme = get_theme()
    return render_template(
        'tfidf.html',
        tfidf_input=user_input,
        tf_results=tf_results,
        relevant_tfidf_scores=relevant_tfidf_scores,
        theme=current_theme
    )

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_post_route():
    user_input = request.form['text_input']
    sentiment_label, sentiment_polarity = analyze_sentiment(user_input, sentiment_model, sentiment_tokenizer, sentiment_max_len)
    current_theme = get_theme()
    return render_template(
        'sentiment.html',
        sentiment_input=user_input,
        sentiment_label=sentiment_label,
        sentiment_polarity=sentiment_polarity,
        theme=current_theme
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)

