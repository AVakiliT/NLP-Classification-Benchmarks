import os
from functools import partial
from multiprocessing import Pool
from os import cpu_count
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils import get_lm, get_laplace, predict_text

# %%
sns.set()
POOL = Pool(cpu_count() // 2)


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
df_train['text_char'] = [i.lower().strip() for i in df_train.text]
df_train.text = [[p.stem(j) for j in word_tokenize(i.lower().strip())] for i in tqdm(df_train.text)]
ls = le.fit_transform(df_train.label)
df_train.label = ls
NUM_CLASSES = len(le.classes_)
# %%
df_train, df_valid = train_test_split(df_train, test_size=1 / NUM_CLASSES, shuffle=True, stratify=df_train.label)


# %%
def train_lms(create, df_train, is_bigram=True, is_char=False, is_mp=False):
    get_lm_p = partial(get_lm, create, df_train, is_bigram, is_char)
    lms = POOL.map(get_lm_p, range(NUM_CLASSES)) if is_mp else list(map(get_lm_p, range(NUM_CLASSES)))
    return lms


def predict(lms, _df_test, is_bigram=True, is_char=False, is_mp=False):
    texts = [(text, text_char, index) for index, (_, text, l, text_char) in enumerate(_df_test.itertuples())]
    predict_text_p = partial(predict_text, lms, NUM_CLASSES, is_bigram, is_char)
    preds = POOL.map(predict_text_p, texts) if is_mp else list(map(predict_text_p, texts))
    return preds


# %%

print("BIGRAM TRAIN")
bigram_lms = train_lms(get_laplace, df_train, is_bigram=True, is_char=False)
print("UNIGRAM TRAIN")
unigram_lms = train_lms(get_laplace, df_train, is_bigram=False, is_char=False)
print("CHAR TRAIN")
char_lms = train_lms(get_laplace, df_train, is_bigram=True, is_char=True)
# %%
print("BIGRAM PREDICT")
bigram_valid_preds = predict(bigram_lms, df_valid, is_bigram=True, is_char=False, is_mp=True)
print("UNIGRAM PREDICT")
unigram_valid_preds = predict(unigram_lms, df_valid, is_bigram=False, is_char=False, is_mp=True)
print("CHAR PREDICT")
char_valid_preds = predict(char_lms, df_valid, is_bigram=True, is_char=True, is_mp=True)


# %%

def metrics(preds, y):
    acc = np.array([preds == y[:len(preds)]]).mean()
    return acc


with open('results.txt', 'w') as f:
    for name, preds in zip(['bi', 'uni', 'char'], [bigram_valid_preds, unigram_valid_preds, char_valid_preds]):
        print(name, metrics(preds, df_valid.label), file=f)
        cm = confusion_matrix(preds, df_valid.label)
        plt.clf()
        ax = sns.heatmap(cm, fmt="d")
        plt.savefig(name + '.png')

# %%

POOL.close()
