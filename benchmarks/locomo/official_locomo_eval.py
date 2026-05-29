"""
VENDORED VERBATIM from snap-research/locomo task_eval/evaluation.py
(https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py).

Only the QA-scoring subset is copied so we score with the EXACT official
metric — no reimplementation. The functions below (normalize_answer,
exact_match_score, f1_score, f1) are byte-for-byte the upstream code; the
heavy bert_score / rouge paths are intentionally omitted (not used for the
QA token-F1 leaderboard). `score_qa` reproduces the per-category dispatch
from the upstream `eval_question_answering` loop (lines 203-221) exactly.

Do not "clean up" this file — divergence from upstream defeats its purpose.
"""

import string
import regex
from collections import Counter

from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return sum(
        max(f1_score(p, gt) for p in predictions) for gt in ground_truths
    ) / len(ground_truths)


def score_qa(prediction, answer, category):
    """Per-category dispatch, VERBATIM from eval_question_answering
    (snap-research/locomo task_eval/evaluation.py, lines 203-221).

      cat 3 : answer = answer.split(';')[0].strip()
      cat 2,3,4 : f1_score(output, answer)
      cat 1     : f1(output, answer)            # multi-hop sub-answer split
      cat 5     : 1 if 'no information available' / 'not mentioned' in output
    """
    output = prediction or ""
    answer = str(answer)
    if category == 3:
        answer = answer.split(';')[0].strip()

    if category in [2, 3, 4]:
        return f1_score(output, answer)
    elif category in [1]:
        return f1(output, answer)
    elif category in [5]:
        if 'no information available' in output.lower() \
                or 'not mentioned' in output.lower():
            return 1.0
        else:
            return 0.0
    else:
        raise ValueError(f"unknown category {category!r}")
