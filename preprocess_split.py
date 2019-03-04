import csv
import os

import nltk
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

#%%

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
df['text_split'] = ['\t'.join([' '.join(word_tokenize(j)) for j in sent_tokenize(i.strip().lower())]) for i in tqdm(df.text)]
df2['text_split'] = ['\t'.join([' '.join(word_tokenize(j)) for j in sent_tokenize(i.strip().lower())]) for i in tqdm(df2.text)]
df.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df.text)]
df2.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df2.text)]
le = LabelEncoder()
ls = le.fit_transform(df.label)
df.label = ls
ls2 = le.transform(df2.label)
df2.label = ls2

df.to_csv('train_clean_split.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
df2.to_csv('test_clean_split.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
