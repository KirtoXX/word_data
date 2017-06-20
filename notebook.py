from keras import metrics
import numpy as np

a = np.array([1,1,0,0,0])
b = np.array([1,0,0,0,0])

result = metrics.categorical_accuracy(a,b)

print(result)