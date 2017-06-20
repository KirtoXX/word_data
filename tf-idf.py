from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy import sparse

#load_data
sentes = []
with open('data/finish_data.txt') as fileTrainRaw:
    for line in fileTrainRaw:
        sentes.append(line)


count_vect = CountVectorizer()
count = count_vect.fit_transform(sentes)


tf_transformer = TfidfTransformer(use_idf=False)
tf_transformer.fit(count)

data = tf_transformer.transform(count)
#print(count_vect.vocabulary_.get('刺鼻'))
#print(count.shape)
data2 = data.todense()

