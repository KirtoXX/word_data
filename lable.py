import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = np.zeros([1800,11])
with open('data/lable2.txt') as text:
    j = 0
    for line in text:
        temp = list(line)
        result = np.zeros([11])
        for i in temp:
            if i=='1':
                result[0]=1
            elif i=='2':
                result[1]=1
            elif i=='3':
                result[2]=1
            elif i=='4':
                result[3]=1
            elif i=='5':
                result[4]=1
            elif i=='6':
                result[5]=1
            elif i=='7':
                result[6]=1
            elif i=='8':
                result[7]=1
            elif i=='9':
                result[8]=1
            elif i=='a':
                result[9]=1
            elif i=='b':
                result[10]=1
        result = result.reshape([1,11])
        data[j] = result
        j+=1

np.save('data/onehotlable.npy',data)
print(data.shape)











