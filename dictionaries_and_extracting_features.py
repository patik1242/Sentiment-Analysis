weak_positive_words = {
    "good","nice","well","fine","pleasant","happy","glad","like","liked","likes",
    "helpful","useful","support","supporting","satisfied","okay","ok","decent",
    "positive","appreciate","enjoy","enjoyed","enjoying","clear","simple","fair",
    "reasonable","improve","improving","improved","safe","stable","smooth",
    "correct","proper","welcome","benefit","beneficial"
}

strong_positive_words = {
    "excellent","amazing","great","fantastic","wonderful","love","loved","awesome",
    "perfect","brilliant","incredible","superb","outstanding","best","successful",
    "success","win","winning","strong","strongly","highly","recommend","recommended"
}

weak_negative_words = {
    "bad","poor","unhappy","dislike","issue","problem","concern","weak","wrong",
    "confusing","unclear","boring","slow","annoyed","tired","worse","low","minor",
    "unsure","uncertain","doubt","complain","complaining","complaint","hard",
    "difficult","negative","lacking","miss","missing","unprepared","unreliable"
}


strong_negative_words = {
    "terrible","horrible","awful","disgusting","hate","worst","failure","fail",
    "broken","useless","worthless","angry","furious","disaster","catastrophe",
    "ruined","ruin","sucks","sucked","hurt","damage","damaging","dangerous"
}


def extract_features(text):
    # Inicjalizacja liczników i cech binarnych
    # (wszystkie wartości startują od 0, co upraszcza późniejsze zliczanie)
    weak_pos = strong_pos = weak_neg = strong_neg = 0
    positive_ratio = negative_ratio = 0
    is_positive_dominant = is_negative_dominant = 0
    has_link = starts_with_RT = digit_count = 0
    has_hashtag = 0

    # Podział tekstu na tokeny
    words = text.split()
    text_len = len(words) if len(words) > 0 else 1  # zabezpieczenie przed dzieleniem przez zero

    # Główna pętla zliczająca obecność słów z kategorii sentymentu
    for w in words:
        token = w.lstrip("#")  # usuwa # aby poprawnie zliczać słowa typu "#good"

        if token in weak_positive_words:
            weak_pos += 1
        elif token in strong_positive_words:
            strong_pos += 1
        elif token in weak_negative_words:
            weak_neg += 1
        elif token in strong_negative_words:
            strong_neg += 1
        elif w.startswith("#"):
            # Oddzielnie wykrywane są wszystkie hashtagi
            has_hashtag = 1

    # Proporcja pozytywnych i negatywnych słów względem długości tekstu
    positive_ratio = (weak_pos + strong_pos) / text_len
    negative_ratio = (weak_neg + strong_neg) / text_len

    # Flagi dominacji sentymentu – wykrywamy wyraźną przewagę jednej strony
    if positive_ratio - negative_ratio > 0.1:
        is_positive_dominant = 1
    if negative_ratio - positive_ratio > 0.1:
        is_negative_dominant = 1

    # Detekcja linków
    if "http" in text.lower():
        has_link = 1

    # Detekcja retweetów (RT ...)
    clean = text.strip().upper()
    if clean.startswith("RT "):
        starts_with_RT = 1

    # Liczenie wszystkich cyfr w tweecie – przydatne do wykrywania dat, czasu, liczb
    for char in text:
        if char.isdigit():
            digit_count += 1

    # Ogólna polaryzacja tekstu (różnica między pozytywnymi/negatywnymi)
    polarity = positive_ratio - negative_ratio

    # Zwrócenie kompletu cech w formacie słownika
    return {
        "weak_pos": weak_pos,
        "strong_pos": strong_pos,
        "weak_neg": weak_neg,
        "strong_neg": strong_neg,
        "is_positive_dominant": is_positive_dominant,
        "is_negative_dominant": is_negative_dominant,
        "word_count": text_len,
        "polarity": polarity,
        "has_link": has_link,
        "starts_with_RT": starts_with_RT,
        "digit_count": digit_count,
        "has_hashtag": has_hashtag
    }
