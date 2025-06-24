# pos.py
import spacy

# Remove direct spaCy model loading here.
# The nlp model will now be passed as an argument from app.py.

def perform_pos_tagging(text, nlp_model):
    """
    Performs Part-of-Speech (POS) tagging on the given text.
    Args:
        text (str): The input text.
        nlp_model: The pre-loaded spaCy language model instance (e.g., from app.py).
    Returns:
        list: A list of dictionaries, each containing 'text', 'pos_', and 'tag_' for a token.
    """
    if nlp_model is None:
        print("POS model (spaCy) not loaded. Cannot perform POS tagging. Ensure app.py loaded it correctly.")
        return [] # Return empty list if model isn't available
    
    doc = nlp_model(text) # Use the passed nlp_model
    # Note: spacy's token.pos_ gives the coarse-grained tag (e.g., "NOUN", "VERB")
    # token.tag_ gives the fine-grained tag (e.g., "NN", "VBP")
    pos_tags = [{"text": token.text, "pos_": token.pos_, "tag_": token.tag_} for token in doc]
    return pos_tags

