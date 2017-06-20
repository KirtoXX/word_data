#coding:utf-8
import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging, gensim, os


sentences = LineSentence('data/corpusSegDone.txt')
'''
doc：http://radimrehurek.com/gensim/models/word2vec.html

cbow_mean = 1 :the hidden layer is mean of context(word)
sg = 0 :cbow type
hs = 1 :hierarchical softmax 
hs = 0 :negative sampling
'''
model = gensim.models.Word2Vec(sentences,sg=0,hs=1,size=10,window=2,min_count=3,workers=4,alpha=0.05,iter=3000)
model.save('weight/word2vector3.model')
print('ok')
a = model.wv.similarity('香味','气味')
b = model.wv.similarity('香味','产品')
print(a)
print(b)

