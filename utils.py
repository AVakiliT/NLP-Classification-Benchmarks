from nltk import bigrams, ngrams, word_tokenize
from nltk.lm import Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten
import numpy as np

def get_lm(create, df_train, is_bigram, is_char, i):
    print(i)
    lm = create()
    df_sub = df_train[df_train.label == i]
    text = df_sub.text_char if is_char else df_sub.text
    text_bigrams = list(map(lambda x: list(bigrams(pad_both_ends(x, n=2))), text))
    text_unigrams = list(map(lambda x: list(ngrams(pad_both_ends(x, n=1), n=1)), text))
    vocab_text = list(flatten(pad_both_ends(sent, n=2) for sent in text))

    if is_bigram:
        lm.fit(text_bigrams, vocab_text)
        lm.fit(text_unigrams)
    else:
        lm.fit(text_unigrams, vocab_text)
    return lm

def get_laplace():
    return Laplace(1)


def predict_text(lms, NUM_CLASSES, is_bigram, is_char, text):
    text, text_char, index = text
    print(index)
    _ngrams = list(ngrams(pad_both_ends(text_char if is_char else text, 2), 2 if is_bigram else 1))
    p = [lms[i].entropy(_ngrams) for i in range(NUM_CLASSES)]
    pred = np.argmin(p)
    return pred

def preprocess(p, text):
    return  [p.stem(j) for j in word_tokenize(i.lower().strip())]