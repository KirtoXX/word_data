from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.engine import Input, Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD
import numpy as np


input = Input(shape=[60,10])

x = Conv1D(nb_filter=128,filter_length=5,activation='relu')(input)
x = MaxPooling1D(5)(x)

x = Conv1D(nb_filter=128,filter_length=5,activation='relu')(x)
x = MaxPooling1D(5)(x)

x = Flatten()(x)

x = Dense(256,name='fc3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(64,name='fc4')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Dense(11,name='fc5')(x)
x = BatchNormalization()(x)

logit = Activation('softmax')(x)

netowrk = Model(input,logit)
print('netowrk build finish')

x = np.load('data/wv_mat3.npy')
y = np.load('data/onehotlable.npy')
print('data load finish')

opt = SGD(lr=0.1,momentum=0.7,decay=0.9)

netowrk.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

netowrk.fit(x,y,batch_size=32,nb_epoch=1000,verbose=1,validation_split=0.2)
netowrk.save_weights('weight/20170326-1.h5')