import re
import sys
import string
import unicodedata
from typing import Optional, List, Union


PUNCT = {chr(i) for i in range(sys.maxunicode)
         if unicodedata.category(chr(i)).startswith('P')}.union(string.punctuation)

NUMERIC_PUNCT = PUNCT.copy()
_ = [NUMERIC_PUNCT.remove(e) for e in [".", ",", "-", "+"]]

CHARACTERS = r"[a-zA-Zа-яА-Я]"

WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar', 'ru']
MIXED_SEGMENTATION_LANGS = ['zh']


def whitespace_tokenize(text: str) -> List[str]:
    """
    Tokenize text with white spaces (default to " ", "\n", "\t")
    Args:
        text: a string;
    Returns:
        a list of tokens;
    """
    return text.split()


def mixed_segmentation(text: str) -> List[str]:
    """
    Tokenize text with white spaces & punctuation characters.
    Args:
        text: a string;
    Returns:
        a list of tokens;
    """
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def remove_punc(text: str, punctuation: Optional[List[str]] = []) -> str:
    return ''.join(ch for ch in text if ch not in punctuation)


def remove_articles(text: str, lang: Optional[str] = "en"):
    """
    Remove articles in different languages for comparing only meaningful texts.
    Args:
        text: a text to clean from articles;
        lang: a language identifier (i.e. en/ru/de ...);
    Returns:
        clean text as string.
    """
    if lang == 'en':
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    elif lang == 'ru':
        return re.sub(r'\b(в|из|к|по|за|у|от|на|под)\b', ' ', text)
    elif lang == 'es':
        return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
    elif lang == 'hi':
        return text # Hindi does not have formal articles
    elif lang == 'vi':
        return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
    elif lang == 'de':
        return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
    elif lang == 'ar':
        return re.sub('\sال^|ال', ' ', text)
    elif lang == 'zh':
        return text # Chinese does not have formal articles
    else:
        raise Exception('Unknown Language {}'.format(lang))


def remove_tailing_characters(text: str) -> str:
    return re.sub(r'[a-zA-Zа-яА-Я ]*\Z', '', text).strip()


def remove_starting_characters(text: str) -> str:
    return re.sub(r'\A[a-zA-Zа-яА-Я ]*', '', text).strip()


def white_space_fix(text, lang: Optional[str] = "en") -> str:
    if lang in WHITESPACE_LANGS:
        tokens = whitespace_tokenize(text)
    elif lang in MIXED_SEGMENTATION_LANGS:
        tokens = mixed_segmentation(text)
    else:
        raise Exception('Unknown Language {}'.format(lang))
    return ' '.join([t for t in tokens if t.strip() != ''])

def lower(text) -> str:
    return text.lower()

def normalize_string_answer(s: str,
                            punktuation: Optional[List[str]] = PUNCT,
                            lang: Optional[str] = "en") -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Args:
        s: string, typically the answer span predicted by a QA model.
        lang: ISO code of language.
        punct: set of punctuation characters.

      Returns:
        string, after applying normalization rules.
    """
    return white_space_fix(remove_articles(remove_punc(lower(s), punktuation), lang), lang)

def normalize_numeric_answer(s,
                             punktuation: Optional[List[str]] = NUMERIC_PUNCT,
                             lang: Optional[str] = "en") -> str:
    """
    Lower text and remove punctuation, all surrounding string characters
    and extra whitespace.
    """
    return white_space_fix(
        remove_tailing_characters(
            remove_starting_characters(
                remove_punc(lower(s), punktuation))), lang)


def check_if_numeric(val: str) -> bool:
    """
    Checks whether a given value is:
    - a digit;
    - an integer number;
    - a floating point number;
    """
    is_number = False
    if isinstance(val, float) or isinstance(val, int):
        return True

    if val.isdigit() or val.isnumeric():
        return True

    try:
        _ = int(val)
        return True
    except:
        try:
            _ = float(val)
            return True
        except Exception as e:
            return False


def convert_to_numeric(val: str) -> Optional[Union[int, float]]:
    """
    Converts a given value to:
    - a digit;
    - an integer number;
    - a floating point number;
    (if it is possible).
    """
    if not check_if_numeric(val):
        return None
    try:
        return int(val)
    except:
        try:
            return float(val)
        except Exception as e:
            return None


def map_prediction_to_answer(prediction: str,
                             answer_options: List[str],
                             default_value: Optional[str] = "-100") -> str:
    """
    Performs direct mapping of string prediction (i.e. decoded) to labels.
    Args:
        prediction: a string prediction (i.e. decoded);
        answer_options: a list fo string available labels;
        default_value: a default value to fill non-valid predictions;

    Returns:
        a single string label.
    """
    if isinstance(prediction, str):
        for answer in answer_options:
            if answer in prediction:
                return answer
    else:
        for answer in answer_options:
            if prediction == answer:
                return answer
    return default_value