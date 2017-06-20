from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging, gensim, os
import numpy as np

model = Word2Vec.load('weight/word2vector2.model')

sentes = []
with open('data/finish_data_without_signl.txt') as fileTrainRaw:
    for line in fileTrainRaw:
        sentes.append(line)

nb_word = 150
nb_vector = 10

mat = np.zeros([1800,nb_word,nb_vector,1])

for i in range(1800):
    count = 0
    mat2 = np.zeros([nb_word, nb_vector])
    s = sentes[i].split()

    for word in s:
        try:
            temp = model.wv[word]
        except:None
        mat2[count] = temp
        count += 1

    #将该句话的mat reshape成3维张量
    mat2 = mat2.reshape(nb_word,nb_vector,1)
    mat[i] = mat2

np.save('data/wv_mat2_without_sig.npy',mat)


