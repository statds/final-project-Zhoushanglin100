import os
import numpy as np
import pandas as pd
import h5py
import sys
import time
import datetime
import subprocess
import keras 
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization

######################### Get Data Directory ########################
data_path = sys.argv[1]
bthsize = int(sys.argv[2])
# Beef --> 30
# Earthquakes --> 32


doc_name = data_path.split('/')[-2]
epoch = 1000
data_list = os.listdir(data_path)  # list all file in this file

########################## Get train datasets ########################

train_list = [each for each in data_list if each.startswith('train_ser')]
total_train = len(train_list)
get_dim = pd.read_table(''.join((data_path, 'train_ser_1')), sep = ' ', header = None)
train_set_rows = get_dim.shape[0]
train_set_cols = get_dim.shape[1]

train_label = np.loadtxt(''.join((data_path, 'train_label')), delimiter = '\n')
train_set = np.ndarray((total_train, train_set_rows, train_set_cols))

for i in range(1, len(train_list)):
	train_set[i] = pd.read_table(''.join((data_path, 'train_ser_', str(i))), sep = ' ', header = None)

######################## Get test datasets #######################

test_list = [each for each in data_list if each.startswith('test_ser')]
total_test = len(test_list)
get_dim = pd.read_table(''.join((data_path, 'test_ser_1')), sep = ' ', header = None)
test_set_rows = get_dim.shape[0]
test_set_cols = get_dim.shape[1]

test_label = np.loadtxt(''.join((data_path, 'test_label')), delimiter = '\n')
test_set = np.ndarray((total_test, test_set_rows, test_set_cols))

for i in range(1, len(test_list)):
	test_set[i] = pd.read_table(''.join((data_path, 'test_ser_', str(i))), sep = ' ', header = None)

########################## Build Model ########################

keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

nb_classes = len(np.unique(np.concatenate([train_label,test_label])))
batch_size = bthsize
nb_epochs = epoch

#### Normalization datasets ####

train_set = train_set[..., np.newaxis]   # reshape
test_set = test_set[..., np.newaxis]     # reshape
# mean = np.mean(train_set)  # mean for data centering
# std = np.std(train_set)  # std for data normalization
# train_set = (train_set - mean)/std
# test_set = (test_set - mean)/std

#### Add Dummy to label (Scale) ####

train_label = train_label[..., np.newaxis]   # reshape
test_label = test_label[..., np.newaxis]     # reshape

train_label = (train_label - train_label.min())/(train_label.max() - train_label.min())*(nb_classes - 1)
y_train = keras.utils.to_categorical(train_label, nb_classes)

test_label = (test_label - test_label.min())/(test_label.max() - test_label.min())*(nb_classes - 1)
y_test = keras.utils.to_categorical(test_label, nb_classes)


#### Start Model ####

x_train = train_set
x_test = test_set

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#########
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
##########

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#######################################################################
 
opt = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(x_test, y_test), shuffle=True)

##### save model #######
# model_name = 'Model_' + str(doc_name) + '_' + str(layer1_size) + '_' + str(layer2_size) + '_' + str(layer3_size) +'_' + str(layer4_size) + '_' + str(layer5_size) +'_'+str(layer1_kernel) + '_' + str(nb_epochs) + '_' + str(time.mktime(datetime.datetime.now().timetuple()))
# model.save(str(model_name) +'.h5')

#Print the testing results which has the lowest training loss.
#log = pd.DataFrame(fit.history)
#print log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc']


