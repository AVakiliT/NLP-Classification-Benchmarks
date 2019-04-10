import csv
import os

import nltk
import pandas as pd
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from matplotlib import pyplot as plt

# %%

def preprocess(root):
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


df = preprocess('C50/C50train')
df2 = preprocess('C50/C50test')
p = nltk.PorterStemmer()
df.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df.text)]
df2.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df2.text)]
le = LabelEncoder()
ls = le.fit_transform(df.label)
df.label = ls
ls2 = le.transform(df2.label)
df2.label = ls2

df.to_csv('train_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
df2.to_csv('test_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# %% AGNEWS
import nltk
import pandas as pd
import re
import csv

tr = 'agnews/train.csv'
te = 'agnews/test.csv'


def get_ag(file):
    df = pd.read_csv(file, header=None)
    f = lambda s: re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", " ").replace('\\', ' ').lower()
    temp_df = [(' '.join(nltk.word_tokenize(f(r[2]) + ' ' + f(r[3]))), r[1] - 1) for r in df.itertuples()]
    new_df = pd.DataFrame.from_records(temp_df)
    return new_df


tr = get_ag(tr)
te = get_ag(te)

# %%
tr.to_csv('agnews/train_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])
te.to_csv('agnews/test_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])

# %% YELP_FULL
import nltk
import pandas as pd
import re
import csv

tr = 'yelp_full/train.csv'
te = 'yelp_full/test.csv'


def get_ag(file):
    df = pd.read_csv(file, header=None)
    f = lambda s: s.lower().replace("\\n", " ").replace('\\', ' ')
    temp_df = [(' '.join(nltk.word_tokenize(f(r[2]))), r[1] - 1) for r in df.itertuples()]
    new_df = pd.DataFrame.from_records(temp_df)
    return new_df


tr = get_ag(tr)
te = get_ag(te)
# %%

tr.to_csv('yelp_full/train_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])
te.to_csv('yelp_full/test_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])

# %%
tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

f = lambda s: s.lower().replace("\\n", " ").replace('\\', ' ')


def get_ng(tr):
    return pd.DataFrame.from_records(
        [(' '.join(nltk.word_tokenize(f(text))[:400]), target) for text, target, in zip(tr.data, tr.target)])


tr = get_ng(tr)
te = get_ng(te)

tr = tr[tr[0].apply(len) > 50]
te = te[te[0].apply(len) > 50]

# %%

tr.to_csv('ng20/train_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])
te.to_csv('ng20/test_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, header=["text", "label"])