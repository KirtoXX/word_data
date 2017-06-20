#coding:utf-8
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging, gensim, os
import numpy as np



model = Word2Vec.load('weight/word2vector3.model')
a = model.wv.similarity('香味','好闻')
b = model.wv.similarity('香味','柔软')

c = model.wv['香味']
d = np.sum(c)

print(c)
print(d)