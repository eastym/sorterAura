from tensorflow.python import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import utils as np_utils
from keras import backend as K
import numpy as np
import matplotlib.pylab as plt




# モデルを作る
Test_NN = Sequential()
Test_NN.add(Dense(4, input_dim=3))

Test_NN.add(Activation('sigmoid'))

Test_NN.add(Dense(3))

Test_NN.add(Activation('softmax'))

Test_NN.summary()

# モデルをコンパイル
Test_NN.compile(optimizer='adam',loss='categorical_crossentropy')

# 学習データ
x, y = prime.get_xy() 

# モデルを学習
Test_NN.fit(x,y,epochs=3)

# モデルをテスト
Test_NN.predict(x)