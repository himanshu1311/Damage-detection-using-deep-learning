from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import uuid
import time
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Conv2DTranspose,UpSampling2D


#data import
path = "D:/Himanshu/data_cc/training/"                       #training data path
vpath= "D:/Himanshu/data_cc/validation/"                     #validation data path



#training data
input_files = [f for f in listdir(path) if isfile(join( path,f)) and f.endswith('.bins') ]
np.random.shuffle(input_files)          # we'll take random set from available data files
xs = np.empty( (0), dtype='float32')    #  input data
ys = np.empty((0,2), dtype='float32')   #  label data
for i in input_files:
    bxs = np.fromfile(path+i, dtype=np.uint16).astype('float32')
    bxs -= bxs.mean()
    bxs /= bxs.std() +0.00001           #avoid division by zero
    xs = np.concatenate((xs,bxs))
    bys = np.loadtxt(path + i[:-5] +'.labels')
    ys = np.concatenate((ys,bys) )

xs = np.reshape(xs, (-1,256,256,1), 'C')


#validation data
vinput_files = [f for f in listdir(vpath) if isfile(join( vpath,f)) and f.endswith('.bins') ]
vxs = np.empty( (0), dtype='float32')    #  input data
vys = np.empty((0,2), dtype='float32')   #  label data
for i in vinput_files:
    vbxs = np.fromfile(vpath+i, dtype=np.uint16).astype('float32')
    vbxs -= vbxs.mean()
    vbxs /= vbxs.std() +0.00001           #avoid division by zero
    vxs = np.concatenate((vxs,vbxs))
    vbys = np.loadtxt(vpath + i[:-5] +'.labels')
    vys = np.concatenate((vys,vbys) )

vxs = np.reshape(vxs, (-1,256,256,1), 'C')


#Pre-trained ResNet50 model
base_model = ResNet50(weights= None, include_top=False, input_shape= (256, 256, 1))

x = base_model.output
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
final = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=final)



# Data augmentation for training
batch_size = 32
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                   width_shift_range=0.2,  # horizontal shift
                                   zoom_range=0.2,  # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   brightness_range=[0.2,0.8])  # brightness



test_datagen = ImageDataGenerator(rotation_range=5,  # rotation
                                  width_shift_range=0.2,  # horizontal shift
                                  zoom_range=0.2,  # zoom
                                  horizontal_flip=True)  # horizontal flip
                                  brightness_range=[0.2,0.8])  # brightness



#Trainnig the model
opt = RMSprop(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
steps_per_epoch= math.ceil(1. * x / BATCH_SIZE)
history = model.fit(train_datagen.flow(xs,ys[:,0]),batch_size=batch_size,epochs=300,
						steps_per_epoch=xs.shape[0] // batch_size, 
						validation_data=(test_datagen.flow(vxs,vys[:,0])),verbose=1)


#Evaluation plot

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()