import nltk

from nltk.lm import MLE, Laplace, NgramCounter, KneserNeyInterpolated
from nltk.lm.api import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.lm.smoothing import _count_non_zero_vals
from nltk.util import bigrams, ngrams
import numpy as np
import math
import operator
import os
from collections import defaultdict, Counter
from functools import reduce, partial

import spacy
import nltk
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.collocations import BigramCollocationFinder

# nlp = spacy.load('en_core_web_sm')
from sklearn import preprocessing

#%%
from nltk import sent_tokenize, word_tokenize
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

cs = []
ls = []
ts = []
root = 'C50/C50test'
for dire in os.listdir(root):
    name = dire.split('-')[-1]
    # topic = dire.split('-')[0]
    course = os.path.join(root, dire)
    for fname in os.listdir(course):
        with open(os.path.join(course, fname)) as f:
            s = f.read()
            # s = [i.replace('\n', ' ').replace('"', '') for i in s if not str.startswith(i, '[')]
            # c.extend(s)
            s = s.replace('\n', ' ')
            cs.append(s)
            ls.append(name)
            # ts.extend([topic] * len(s))
    # cs.extend(c)
    # ls.extend(l)


df = pd.DataFrame(dict(text=cs, label=ls,
                       # topic=ts
                       ))


# %%
datasets = []
for i, filename in enumerate(os.listdir('stem')):
    with open(os.path.join('stem', filename), 'r') as f:
        for line in f:
            # print(line)
            datasets.append(dict(text=line.strip().split(' '), label=i))

df_old = pd.DataFrame(datasets)


#%%
p = nltk.PorterStemmer()
df.text = [[p.stem(j) for j in word_tokenize(i.lower().strip())] for i in tqdm(df.text)]
le = LabelEncoder()
ls = le.fit_transform(df.label)
df.label = ls
#%%
df_train, df_test = train_test_split(df, test_size=1/50, shuffle=True)

#%%
def create():
    return Laplace(2)

#%%
lms = []
for i in range(len(le.classes_)):
    lm = create()
    df_sub = df_train[df_train.label==i]
    text_bigrams = list(map(lambda x: list(bigrams(pad_both_ends(x, n=2))), df_sub.text))
    text_unigrams = list(map(lambda x: list(ngrams(pad_both_ends(x, n=1), n=1)), df_sub.text))
    print(len(text_bigrams))
    vocab_text = list(flatten(pad_both_ends(sent, n=2) for sent in df_sub.text))
    lm.fit(text_bigrams, vocab_text)
    lm.fit(text_unigrams)
    lms.append(lm)

#%%
preds = []
for _, text, l in tqdm(df_test.itertuples(), total=len(df_test)):
    text_bigrams = list(bigrams(pad_both_ends(text, n=2)))
    p = [lms[i].entropy(text_bigrams) for i in range(11)]
    pred = np.argmin(p)
    preds.append(pred)

print(np.array([preds == df_test.label[:len(preds)]]).mean())
#%%
for i in range(16):
    df_sub = df_test[df_test.label==i]
    xx = [list(bigrams(pad_both_ends(x, n=2))) for x in df_sub.text]


#%%
class MyKneser(LanguageModel):

    def __init__(self, order, vocabulary=None, counter=None):
        super().__init__(order, vocabulary, counter)

        self.kn = {word: sum(1 for i in lm.counts[2] if word in lm.counts[i]) for word in tqdm(lm.vocab.counts.keys())}
        self.kn_sum = sum(self.kn.values())

    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        return self.alpha(word, prefix_counts), self.gamma(prefix_counts)

    def alpha(self, word, prefix_counts):
        return max(prefix_counts[word] - self.discount, 0.0) / prefix_counts.N()

    def gamma(self, prefix_counts):
        return self.discount * _count_non_zero_vals(prefix_counts) / prefix_counts.N()

    def unmasked_score(self, word, context=None):
        prefix_counts = self.counts[context]
        kn = self.kn[word] / self.kn_sum
        return self.alpha(word, prefix_counts) + self.gamma(prefix_counts) * kn
