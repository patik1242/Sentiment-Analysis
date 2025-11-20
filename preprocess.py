import re

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9#@\s]", "", text)
    text = text.split()
    text = " ".join(text)
    text = text.strip()
    return text


