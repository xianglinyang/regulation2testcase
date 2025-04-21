import logging

def load_regulation_text(filepath):
    # txt file
    logging.info(f"Loading regulation text from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()