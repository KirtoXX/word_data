import numpy as np

data = np.load('wv_mat.npy')
data = data.reshape([1800, 60, 10])

np.save('wv_mat3.npy',data)