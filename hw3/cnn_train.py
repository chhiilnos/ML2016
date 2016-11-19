import pickle
import numpy as np
import sys
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

# batch_size, nb_classes, nb_epoch, data_augmentation
batch_size = 500
nb_classes = 10
nb_epoch = 60
nb_epoch_self = 20
confident = 0.96
rounds = 10
X_train_unlabel_added_id = []
X_train_unlabel_not_added_id = [i for i in range(45000)]
Y_train_unlabel_added = np.zeros(shape=(0,10))


print(Y_train_unlabel_added.shape)
# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

#import pprint, pickle
# the data, shuffled and split between train and test sets
## labeled data x and y

all_label = pickle.load(open(sys.argv[1]+'all_label.p','rb'))
all_label = np.asarray(all_label)
X_train_label = all_label.reshape((5000,3,32,32))
y_train_label = []
for i in range (10):
  for k in range (500):
    y_train_label.append(i)
y_train_label = np.asarray(y_train_label)
## unlabeled data x and y
all_unlabel = pickle.load(open(sys.argv[1]+'all_unlabel.p','rb'))
all_unlabel = np.asarray(all_unlabel)
X_train_unlabel = all_unlabel.reshape((45000,3,32,32))

test = pickle.load(open(sys.argv[1]+'test.p','rb'))
X_test = np.array(test['data'])
X_test = X_test.reshape((10000,3,32,32))

# convert x to be in range[0,1)
X_train_label = X_train_label.astype('float32')
X_train_unlabel = X_train_unlabel.astype('float32')
X_test = X_test.astype('float32')
X_train_label /= 255
X_train_unlabel /=255
X_test /= 255

# convert class vectors to binary class matrices
Y_train_label = np_utils.to_categorical(y_train_label, nb_classes)
# building model
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',input_shape=(3,32,32)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#fitting
np.random.seed(0)
np.random.shuffle(X_train_label)
np.random.seed(0)
np.random.shuffle(Y_train_label)

model.fit(X_train_label, Y_train_label,
        batch_size=batch_size,
        nb_epoch=nb_epoch,shuffle=True)

#self training
for r in range(rounds):
    if(len(X_train_unlabel_not_added_id)>10):
      #self learning confident
      prob_of_X_train_unlabel_not_added = model.predict_proba(X_train_unlabel[np.ix_(X_train_unlabel_not_added_id)])
      max_prob_of_X_train_unlabel_not_added = np.amax(prob_of_X_train_unlabel_not_added , axis = 1)

      delete_index = []
      for i in range (len(X_train_unlabel_not_added_id)):
        if(max_prob_of_X_train_unlabel_not_added[i] > confident):
          X_train_unlabel_added_id.append(X_train_unlabel_not_added_id[i])
          delete_index.append(i)
      if(len(delete_index)>0):
        Y_train_unlabel_added = np.vstack((Y_train_unlabel_added,np_utils.to_categorical
                                (np.argmax(prob_of_X_train_unlabel_not_added[np.ix_(delete_index,
                                [j for j in range(10)])],axis=1), nb_classes)))
      X_train_unlabel_not_added_id = np.delete(X_train_unlabel_not_added_id,delete_index)

    #fitting
    model.fit(np.concatenate((X_train_label,X_train_unlabel[np.ix_(X_train_unlabel_added_id,
                  [a for a in range(3)],[b for b in range(32)],[c for c in range(32)])]),axis=0),
                  np.vstack((Y_train_label,Y_train_unlabel_added)),
                  batch_size=batch_size,
                  nb_epoch=nb_epoch_self)

model.save(sys.argv[2])









