# Copyright 2022 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for Question Answering (QA) evaluation.
Matches results on the SQuAD (v1.1) and TriviaQA (v1.0) evaluation scripts.
"""

import re
import string
import numpy as np
import collections
from romashka.logging_handler import get_logger

# Set up logging
logger = get_logger(
    name="train",
    logging_level="INFO"
)


def _normalize_answer(text, punc_chars, punc_repl):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def normalize_trivia_qa(answer):
    """
    Normalization used in official TriviaQA evaluation script.
    """
    return _normalize_answer(
        answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
    """
    Normalization used in official SQuAD evaluation script.
    """
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
    """
    Computes the maximum of the metric over all ground truths.
    """
    return max(
        metric_fn(ground_truth, prediction) for ground_truth in ground_truths
    )


def _exact_match_score(target, prediction):
    # if (target[0] in prediction) or (prediction in target[0]):
    #    return True
    # else:
    return target == prediction


def _f1_score(target, prediction):
    """
    Computes token f1 score for a single target and prediction.
    """
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_metrics(targets, predictions):
    """
    Computes exact match and f1 QA scores, expecting pre-normalized text.
    """
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")
    em_list = []

    for p, t in zip(predictions, targets):
        if (t[0] in p and p != ""):
            p = t[0]
        elif (p in t[0] and p != ""):
            p = t[0]
        em_list.append(_metric_max_over_ground_truths(_exact_match_score, t, p))
    em = np.mean(em_list)

    f1_list = []
    for p, t in zip(predictions, targets):
        if (t[0] in p and p != ""):
            p = t[0]
        elif (p in t[0] and p != ""):
            p = t[0]
        f1_list.append(_metric_max_over_ground_truths(_f1_score, t, p))
    f1 = np.mean(f1_list)

    em *= 100
    f1 *= 100
    logger.info("EM = %.2f, F1 = %.2f", em, f1)
    return {"em": em, "f1": f1}


def normalize_mlqa(s, lang, punct, cot=False, is_choose=False):
    """
    Lower text and remove punctuation, articles and extra whitespace.

    Based on third_party/py/xtreme/third_party/evaluate_mlqa.py
    Args:
    s: string, typically the answer span predicted by a QA model.
    lang: ISO code of language.
    punct: set of punctuation characters.

    Returns:
    string, after applying normalization rules.
    """

    whitespace_langs = ['en', 'es', 'hi', 'vi', 'de', 'ar', 'ru']
    mixed_segmentation_langs = ['zh']

    def drop_special(text):
        return re.sub("<extra_id_\d>", "", text)

    def whitespace_tokenize(text):
        return text.split()

    def mixed_segmentation(text):
        segs_out = []

        temp_str = ''
        for char in text:
            if re.search(r'[\u4e00-\u9fa5]', char) or char in punct:
                if temp_str != '':
                    ss = whitespace_tokenize(temp_str)
                    segs_out.extend(ss)
                    temp_str = ''
                segs_out.append(char)
            else:
                temp_str += char
        if temp_str != '':
            ss = whitespace_tokenize(temp_str)
            segs_out.extend(ss)
        return segs_out

    def drop_extras(text):
        return re.sub(r'extraid\d\d*\d*', '', text).strip()

    def drop_articles(text, lang):
        if lang == 'ru':
            return re.sub(r'\b(в|из|к|по|за|у|от|на|под)\b', ' ', text)
        if lang == 'en':
            return text  # re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(
                r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b',
                ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'zh':
            return text

    def white_space_fix(text, lang):
        if lang in whitespace_langs:
            tokens = whitespace_tokenize(text)
        elif lang in mixed_segmentation_langs:
            tokens = mixed_segmentation(text)
        return ' '.join([t for t in tokens if t.strip()])

    def drop_punc(text):
        return ''.join(c for c in text if c not in punct)

    def drop_cot(text):
        result = re.findall('\(.*?\)', text)
        if (len(result) > 0):
            pred = result[0]
            label = re.search('[abcde]', pred)
            if (label != None):
                return pred[label.span()[0]]
        answer_span = re.search("answer", text)
        if (answer_span != None):
            label = re.search('[abcde]', text[answer_span.span()[1]:])
            if (label != None):
                return text[answer_span.span()[1]:]
        return text

    def find_label(text):
        if (text != ''):
            label = re.search('[abcde]', text.split()[0])
            if (label != None):
                return text[label.span()[0]]
        return text

    def drop_answers(text):
        text = text.replace('answer', '')
        text = text.replace('ответ', '')
        return text.strip()

    s = s.lower()
    s = drop_punc(s)
    s = drop_articles(s, lang)
    s = drop_extras(s)

    if is_choose == True:
        s = find_label(s)

    if cot == True:
        s = drop_cot(s)

    s = white_space_fix(s, lang)
    s = drop_answers(s)
    s = drop_special(s)
    return s