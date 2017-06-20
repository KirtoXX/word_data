import gensim
from gensim.models.word2vec import LineSentence

#----read data------
sentences = LineSentence('data/corpusSegDone.txt')


#----build model-------
model = gensim.models.Doc2Vec(min_count=1, window=10, size=100, sample=1e-3, negative=5, workers=3)
model.build_vocab()