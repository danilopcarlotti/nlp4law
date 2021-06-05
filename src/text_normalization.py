from nltk.tokenize import RegexpTokenizer
import nltk


def remove_accents(text):
    accents = {
        "Á": "A",
        "Ã": "A",
        "À": "A",
        "á": "a",
        "ã": "a",
        "à": "a",
        "É": "E",
        "é": "e",
        "Ê": "E",
        "ê": "e",
        "Í": "I",
        "í": "i",
        "Ó": "O",
        "ó": "o",
        "Õ": "O",
        "õ": "o",
        "Ô": "O",
        "ô": "o",
        "Ú": "U",
        "ú": "u",
        ";": "",
        ",": "",
        "/": "",
        "\\": "",
        "{": "",
        "}": "",
        "(": "",
        ")": "",
        "-": "",
        "_": "",
        "Ç": "C",
        "ç": "c",
    }
    text = str(text)
    for k, v in accents.items():
        text = text.replace(k, v)
    return text


def normalize_texts(texts, to_stem=False):
    if to_stem:
        stemmer = nltk.stem.RSLPStemmer()
    normal_texts = []
    tk = RegexpTokenizer(r"\w+")
    stopwords = set(nltk.corpus.stopwords.words("portuguese"))
    for t in texts:
        raw_text = remove_accents(t.lower())  # steps 1 and 2
        tokens = tk.tokenize(raw_text)  # step 3
        processed_text = ""
        for tkn in tokens:
            if tkn.isalpha() and tkn not in stopwords and len(tkn) > 3:  # step 4
                if to_stem:
                    tkn = stemmer.stem(tkn)  # step 5
                processed_text += tkn + " "
        normal_texts.append(processed_text[:-1])
    return normal_texts
