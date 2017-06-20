from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.engine import Input, Model
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import metrics
import numpy as np


input = Input(shape=[150,10,1])

x = Conv2D(nb_filter=256,nb_row=1,nb_col=10,activation='relu',name='conv1')(input)
x = MaxPooling2D(pool_size=[150,1],name='pool1')(x)

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

x = np.load('data/wv_mat2_without_sig.npy')
y = np.load('data/onehotlable.npy')
print('data load finish')

opt = SGD(lr=0.1,momentum=0.7,decay=0.9)

netowrk.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=[metrics.recall])

netowrk.fit(x,y,batch_size=32,nb_epoch=500,verbose=1,validation_split=0.2)
netowrk.save_weights('weight/20170328-1.h5')
