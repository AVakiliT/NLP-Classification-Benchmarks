import numpy as np
from nltk import bigrams, ngrams
from nltk.lm import Laplace
from nltk.lm.models import InterpolatedLanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.lm.smoothing import KneserNey


# TODO fix start and end tokens
def train_lm(create, df_train, is_bigram, is_char, i):
    print(i, '/ 50')
    lm = create()
    df_sub = df_train[df_train.label == i]
    text = df_sub.text_char if is_char else df_sub.text
    text_bigrams = list(map(lambda x: list(bigrams(pad_both_ends(x, n=2))), text))
    text_unigrams = list(map(lambda x: list(ngrams(pad_both_ends(x, n=1), n=1)), text))
    vocab_text = list(flatten(pad_both_ends(sent, n=2 if is_bigram else 1) for sent in text))

    if is_bigram:
        lm.fit(text_bigrams, vocab_text)
        lm.fit(text_unigrams)
    else:
        lm.fit(text_unigrams, vocab_text)
    return lm


def instatiate_laplace():
    return Laplace(1)


def instantiate_abs():
    return AbsInterpolated(2, 0.75)


def predict_text(lms, num_classes, is_bigram, is_char, text):
    text, text_char, index = text
    print(index, '/ 100')
    _ngrams = list(ngrams(pad_both_ends(text_char if is_char else text, 2 if is_bigram else 1), 2 if is_bigram else 1))
    p = [lms[i].entropy(_ngrams) for i in range(num_classes)]
    pred = np.argmin(p)
    return pred


class Abs(KneserNey):
    def alpha(self, word, prefix_counts):
        return 0 if prefix_counts.N() == 0 else (max(prefix_counts[word] - self.discount, 0.0) / prefix_counts.N())

    def gamma(self, prefix_counts):
        return 0.5


class AbsInterpolated(InterpolatedLanguageModel):
    def __init__(self, order, discount=0.1, **kwargs):
        super(AbsInterpolated, self).__init__(
            Abs, order, params={"discount": discount}, **kwargs
        )
