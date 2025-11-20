import re

def preprocess(text):
    # Zabezpieczenie — jeśli dane nie są tekstem (np. NaN), zwracamy pusty string
    if not isinstance(text, str):
        return ""

    # Normalizacja: konwersja do małych liter (redukuje liczbę unikalnych tokenów)
    text = text.lower()

    # Usunięcie wszystkich znaków poza literami, cyframi, spacją i '#'
    # Pozostawienie '#' jest celowe — hashtagi stanowią znaczącą informację w tweetach
    text = re.sub(r"[^a-zA-Z0-9#\s]", "", text)

    # Tokenizacja i usunięcie wielokrotnych spacji
    text = text.split()

    # Połączenie tokenów w oczyszczony string
    text = " ".join(text)

    # Ostateczne usunięcie spacji na początku/końcu
    return text.strip()
