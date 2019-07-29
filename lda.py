from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import csv
import json
import random

import nltk
import pandas as pd

df = pd.read_csv('ng20/train_clean.csv')
sents = df.text

# %%

from gensim import models, corpora

# %%
sents = [x.split() for x in sents]
dictionary = corpora.Dictionary(sents)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in sents]

# %%
num_topic = 12
lda = models.LdaMulticore(doc_term_matrix, num_topics=num_topic, id2word=dictionary,
                          passes=20, chunksize=2000, random_state=3, workers=8)

#%%
import pyLDAvis.gensim
from matplotlib import pyplot as plt
vis = pyLDAvis.gensim.prepare(lda, doc_term_matrix, dictionary=lda.id2word)
pyLDAvis.save_html(vis, 'ng20/lda.html')

#%%
coherence_model_lda = models.CoherenceModel(model=lda, texts=sents, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(coherence_lda)
print(coherence_model_lda.get_coherence_per_topic())


#%%
def doc_freq(w):
    count = 0
    for i in doc_term_matrix:
        for j in i:
            if w == j[0]:
                count += 1
                break
    return count

def doc_freq2(w, v):
    count = 0
    for i in doc_term_matrix:
        words = [j[0] for j in i]
        if w in words and v in words:
            count += 1
    return count


