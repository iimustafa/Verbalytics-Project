# name_entity.py
import spacy

def perform_ner(text, nlp_model):
    """
    Performs Named Entity Recognition (NER) on the given text.
    Args:
        text (str): The input text.
        nlp_model: The pre-loaded spaCy language model instance (e.g., from app.py).
    Returns:
        list: A list of dictionaries, each containing 'text' and 'label' for an entity.
    """
    if nlp_model is None:
        print("NER model (spaCy) not loaded. Cannot perform NER. Ensure app.py loaded it correctly.")
        return [] # Return empty list if model isn't available
    
    doc = nlp_model(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities

