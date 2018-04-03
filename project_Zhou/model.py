import os
import numpy as np
import pandas as pd
import h5py
import sys
import time
import datetime
import subprocess
import keras 
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout

######################### Get Data Directory ########################
data_path = sys.argv[1]
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


######################## Get test datasets #######################

test_list = [each for each in data_list if each.startswith('test_ser')]
total_test = len(test_list)
get_dim = pd.read_table(''.join((data_path, 'test_ser_1')), sep = ' ', header = None)
test_set_rows = get_dim.shape[0]
test_set_cols = get_dim.shape[1]

test_label = np.loadtxt(''.join((data_path, 'test_label')), delimiter = '\n')
test_set = np.ndarray((total_test, test_set_rows, test_set_cols))


########################## Build Model ########################

keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

nb_classes = len(np.unique(np.concatenate([train_label,test_label])))
batch_size = 32
nb_epochs = epoch

train_set = train_set[..., np.newaxis]
test_set = test_set[..., np.newaxis]

#### Normalization datasets ####
train_set = train_set[..., np.newaxis]
test_set = test_set[..., np.newaxis]
mean = np.mean(train_set)  # mean for data centering
std = np.std(train_set)  # std for data normalization
train_set = (train_set - mean)/std
test_set = (test_set - mean)/std

#### Add Dummy to label ####
# tn_label = train_label
# tt_label = test_label
# train_label = (train_label - train_label.min())/(train_label.max() - train_label.min())*(nb_classes - 1)
# y_train = np_utils.to_categorical(train_label, nb_classes)
# test_label = (test_label - test_label.min())/(test_label.max() - test_label.min())*(nb_classes - 1)
# y_test = np_utils.to_categorical(test_label, nb_classes)

#### Start Model ####
x_train = train_set.reshape(train_set.shape[0], train_set.shape[1]*train_set.shape[2], 1, 1)
x_test = test_set.reshape(test_set.shape[0], test_set.shape[1]*test_set.shape[2], 1, 1)




inputs = keras.layers.Input(x_train.shape[1:])

conv0 = keras.layers.Conv2D(128, 48, padding='same')(inputs)
conv0 = keras.layers.normalization.BatchNormalization()(conv0)
conv0 = keras.layers.Activation('relu')(conv0)

conv0 = keras.layers.Conv2D(256, 8, padding='same')(conv0)
conv0 = keras.layers.normalization.BatchNormalization()(conv0)
conv0 = keras.layers.Activation('relu')(conv0)

conv0 = keras.layers.Conv2D(128, 4, padding='same')(conv0)
conv0 = keras.layers.normalization.BatchNormalization()(conv0)
conv0 = keras.layers.Activation('relu')(conv0)

full = keras.layers.pooling.GlobalAveragePooling2D()(conv0)    
out = keras.layers.Dense(nb_classes, activation='softmax')(full)

#conv1 = keras.layers.Conv2D(layer1_size, layer1_kernel)(conv0)
#conv1 = keras.layers.Activation('relu')(conv1)
#conv1 = keras.layers.Conv2D(layer1_size, layer1_kernel)(conv1)
#conv1 = keras.layers.Activation('relu')(conv1)
#conv1 = keras.layers.Conv2D(layer1_size, layer1_kernel)(conv1)
#conv1 = keras.layers.Activation('relu')(conv1)
#conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# 
#conv2 = keras.layers.Conv2D(layer2_size, layer2_kernel)(conv1)
#conv2 = keras.layers.Activation('relu')(conv2)
#conv2 = Flatten()(conv2)
#
#conv3 = keras.layers.Dense(layer3_size, activation='relu')(conv2)
#conv3 = Dropout(0.5)(conv3)
#
#conv4 = keras.layers.Dense(layer4_size, activation='relu')(conv3)
#conv4 = Dropout(0.5)(conv4)
#conv4 = keras.layers.Dense(layer4_size, activation='relu')(conv4)
#conv4 = Dropout(0.5)(conv4)
#
#conv5 = keras.layers.Dense(layer5_size, activation='relu')(conv4)

#out = keras.layers.Dense(nb_classes, activation='softmax')(conv5)

#conv3 = keras.layers.Conv2D(layer3_size, layer3_kernel)(conv2)
##conv3 = keras.layers.normalization.BatchNormalization()(conv3)
#conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#conv3 = keras.layers.Activation('relu')(conv3)
#conv3 = Dropout(0.5)(conv3)

# conv4 = keras.layers.Conv2D(layer4_size, layer4_kernel)(conv3)
# #conv4 = keras.layers.normalization.BatchNormalization()(conv4)
# #conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
# conv4 = keras.layers.Activation('relu')(conv4)
# #conv4 = Dropout(0.5)(conv4)
# 
# conv5 = keras.layers.Conv2D(layer5_size, layer5_kernel, padding='same')(conv4)
# #conv5 = keras.layers.normalization.BatchNormalization()(conv5)
# #conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)
# conv5 = keras.layers.Activation('relu')(conv5)
# #conv5 = Dropout(0.5)(conv5)

# conv6 = keras.layers.Conv2D(64, layer1_kernel, padding='same')(conv5)
# #conv1 = keras.layers.normalization.BatchNormalization()(conv1)
# conv6 = MaxPooling2D(pool_size=(2, 2))(conv6)
# conv6 = keras.layers.Activation('relu')(conv6)

# #full = keras.layers.pooling.GlobalMaxPooling2D()(conv5)  
# full = Flatten()(conv3)
# #full = Dropout(0.5)(full)
# out = keras.layers.Dense(nb_classes, activation='softmax')(full)

model = Model(input=inputs, output=out)
 
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
 
###################### define callback functions ####################
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.5, patience=50, min_lr=0.00001) 
check_p = ModelCheckpoint('tmp', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=100)
tboard = TensorBoard(log_dir='./logs', histogram_freq=100, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=1000, verbose=0, mode='auto')

fit = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=[tboard], validation_data=(x_test,y_test))

##### save model #######
# model_name = 'Model_' + str(doc_name) + '_' + str(layer1_size) + '_' + str(layer2_size) + '_' + str(layer3_size) +'_' + str(layer4_size) + '_' + str(layer5_size) +'_'+str(layer1_kernel) + '_' + str(nb_epochs) + '_' + str(time.mktime(datetime.datetime.now().timetuple()))
# model.save(str(model_name) +'.h5')

#Print the testing results which has the lowest training loss.
#log = pd.DataFrame(fit.history)
#print log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc']


