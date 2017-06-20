from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import numpy as np


X = np.load('data/my_data.npy')
y = np.load('data/onehotlable.npy')
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=0)
print('load data finish')

clf = GradientBoostingClassifier(n_estimators=55, learning_rate=0.1,max_depth=5,subsample=0.6,min_samples_split=0.2)
clf2 = OneVsRestClassifier(clf)
clf2.fit(X_train,y_train)
print('train acc: ',clf2.score(X_train,y_train))
print('test acc:',clf2.score(X_test,y_test))
