import os
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.lm import Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.util import bigrams, ngrams
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from multiprocess import Pool, Process



sns.set()


def get_data(root):
    cs = []
    ls = []
    for dire in os.listdir(root):
        name = dire.split('-')[-1]

        course = os.path.join(root, dire)
        for fname in os.listdir(course):
            with open(os.path.join(course, fname)) as f:
                s = f.read()
                s = s.replace('\n', ' ')
                cs.append(s)
                ls.append(name)

    df = pd.DataFrame(dict(text=cs, label=ls))
    return df


df_train = get_data('C50/C50train')


# %%
p = nltk.PorterStemmer()
le = LabelEncoder()
df_train['text_char'] = [i.lower().strip() for i in tqdm(df_train.text)]
df_train.text = [[p.stem(j) for j in word_tokenize(i.lower().strip())] for i in tqdm(df_train.text)]
ls = le.fit_transform(df_train.label)
df_train.label = ls
# %%
df_train, df_valid = train_test_split(df_train, test_size=1 / 50, shuffle=True, stratify=df_train.label)


# %%
def train_lm(create, _df_train, is_bigram=True, is_char=False):
    lms = []
    for i in tqdm(range(len(le.classes_))):
        lm = create()
        df_sub = _df_train[_df_train.label == i]
        text = df_sub.text_char if is_char else df_sub.text

        if is_bigram:
            text_bigrams = list(map(lambda x: list(bigrams(pad_both_ends(x, n=2))), text))
        text_unigrams = list(map(lambda x: list(ngrams(pad_both_ends(x, n=1), n=1)), text))
        vocab_text = list(flatten(pad_both_ends(sent, n=2) for sent in text))
        lm.fit(text_unigrams, vocab_text)
        if is_bigram:
            lm.fit(text_bigrams)
        lms.append(lm)
    return lms


def predict(lms, _df_test, is_char=False):
    preds = []
    for _, text, l, text_char in tqdm(_df_test.itertuples(), total=len(_df_test)):
        text_bigrams = list(bigrams(pad_both_ends(text_char if is_char else text, n=2)))
        p = [lms[i].entropy(text_bigrams) for i in range(11)]
        pred = np.argmin(p)
        preds.append(pred)

    return preds


# %%
def metrics(preds, y):
    acc = np.array([preds == y[:len(preds)]]).mean()
    return acc


# %%
bigram_valid_preds = predict(bigram_lms, df_valid)
unigram_valid_preds = predict(unigram_lms, df_valid)
char_valid_preds = predict(unigram_lms, df_valid, is_char=True)

# %%
for preds in [bigram_valid_preds, unigram_valid_preds, char_valid_preds]:
    print(metrics(preds, df_valid.label))
    cm = confusion_matrix(preds, df_valid.label)
    ax = sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

# %%


