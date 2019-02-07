#%%
import math
import operator
from collections import defaultdict, Counter
from functools import reduce, partial

import spacy
import nltk
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.collocations import BigramCollocationFinder


nlp = spacy.load('en_core_web_sm')
from sklearn import preprocessing

#%%
dataset = []
with open('./corpus') as f:
    for line in f:
        line = line.split(' ', maxsplit=1)
        dataset.append(dict(text=line[1].lower().strip(), label = 0 if line[0] == '__label__1' else 1))

df = pd.DataFrame(dataset)

#%%
# def tokenize(x):
#     return [t for t in nlp(x)]

def tokenize(x):
    return x.split(' ')

temp = [tokenize(i) for i in tqdm(df.text)]
df.text = temp

#%%

df_train, df_test = train_test_split(df, test_size=0.1)

#%%
def vectorize_corpus_fit_transform(corpus):
    vocab = {
        '<sos>': 0,
        '<eos>': 1,
        '<unk>': 2
    }
    current = 3
    def get_id(token):
        nonlocal current
        if token not in vocab:
            vocab[token] = current
            current += 1
        return vocab[token]

    return [[get_id(i) for i in sent] for sent in corpus], vocab

def vectorize_corpus_transform(corpus, vocab):
    def get_id(token):
        if token not in vocab:
            return vocab['<unk>']
        return vocab[token]

    return [[get_id(i) for i in sent] for sent in corpus], vocab


#%%

def find_ngrams(input_list, n=2):
    pad = max((n-1), 1)
    input_list = [0] * pad + input_list + [1] * pad
    return zip(*[input_list[i:] for i in range(n)])

#%%

def count_ngrams(doc, n=2):
    return Counter(find_ngrams(doc, n))

def count_ngrams_all(corpus, n=2):
    return reduce(operator.add, map(lambda x: count_ngrams(x, n), corpus))

# def count_ngrams_all(corpus, n=2):
#     return reduce(operator.add, [count_ngrams(x, n) for x in tqdm(corpus)])
#%%
tokens, vocab = vectorize_corpus_fit_transform(df_train.text)

bigrams = count_ngrams_all(tokens, 2)

#%%

unigrams = count_ngrams_all(tokens, 1)

#%%

def to_token_ids(tokens):
    return  tuple([vocab[i] for i in tokens])

class LM:
    def __init__(self, ngram_counts, n_1gram_counts, v, vocab) -> None:
        super().__init__()
        self.vocab = vocab
        self.v = v
        self.ngram_counts = ngram_counts
        self.n_1gram_counts = n_1gram_counts

    # def _p(self, posterior, prior):
    #     return (self.n_1gram_counts[posterior] + 1) / (self.n_1gram_counts[prior] + self.v)

    def p(self, prior, next_token):
        posterior = (prior, next_token)
        return (self.ngram_counts[to_token_ids(posterior)] + 1) / (self.n_1gram_counts[to_token_ids((prior,))] + self.v)

    def _p(self, prior, next_token):
        posterior = (prior, next_token)
        return (self.ngram_counts[posterior] + 1) / (self.n_1gram_counts[(prior,)] + self.v)

lm = LM(bigrams, unigrams, len(unigrams), vocab)


#%%
tokens, vocab = vectorize_corpus_transform(df_test.text, vocab)

bigrams_test = count_ngrams_all(tokens, 2)
unigrams = count_ngrams_all(tokens, 1)


#%%
def perprexity(lm, text_tokens):
    bigrams = find_ngrams(text_tokens, 2)
    ps = [math.log(lm._p(b[0], b[1])) for b in bigrams]
    return math.e ** (sum(ps) / len(ps))


