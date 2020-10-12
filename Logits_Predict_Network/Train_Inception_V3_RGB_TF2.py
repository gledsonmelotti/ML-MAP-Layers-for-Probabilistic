import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

tf.get_logger().setLevel('ERROR')
print(tf.__version__)

print("tensorflow {}".format(tf.__version__))

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, Add, Lambda, GaussianDropout
from tensorflow.keras.layers import Average
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.regularizers import l2

import numpy as np

from tqdm import tqdm


import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from itertools import cycle

import scipy.io as sio

## Train

train_dir='D:/RGB/Tres_classes_2/train'
validation_dir='D:/RGB/Tres_classes_2/validation'
test_dir='D:/RGB/Tres_classes_2/test'

train_Pedestrian_dir='D:/RGB/Tres_classes_2/train/Pedestrian'
train_Car_dir='D:/RGB/Tres_classes_2/train/Car'
train_Cyclist_dir='D:/RGB/Tres_classes_2/train/Cyclist'

validation_Pedestrian_dir='D:/RGB/Tres_classes_2/validation/Pedestrian'
validation_Car_dir='D:/RGB/Tres_classes_2/validation/Car'
validation_Cyclist_dir='D:/RGB/Tres_classes_2/validation/Cyclist'

test_Pedestrian_dir='D:/RGB/Tres_classes_2/test/Pedestrian'
test_Car_dir='D:/RGB/Tres_classes_2/test/Car'
test_Cyclist_dir='D:/RGB/Tres_classes_2/test/Cyclist'

###############################################################################
nb_train_samples=len(os.listdir(train_Pedestrian_dir)) + len(os.listdir(train_Car_dir)) + len(os.listdir(train_Cyclist_dir)) 
nb_validation_samples=len(os.listdir(validation_Pedestrian_dir)) + len(os.listdir(validation_Car_dir)) + len(os.listdir(validation_Cyclist_dir))
nb_test_samples = len(os.listdir(test_Pedestrian_dir)) + len(os.listdir(test_Car_dir)) + len(os.listdir(test_Cyclist_dir))
print('######################################################################')
print('nb_train_samples')
print(nb_train_samples)
print('######################################################################')
print('nb_validation_samples')
print(nb_validation_samples)
print('######################################################################')
print('nb_test_samples')
print(nb_test_samples)
print('######################################################################')

batch_size=32
num_epochs=100
DROPOUT = 0.5
num_classes = 3

img_width, img_height, channel_axis = 299, 299, 3

def Convolution2D_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Convolution2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=True,
        kernel_regularizer=l2(1e-4),
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x
    
# Determine proper input shape

model_input = Input(shape = (img_width, img_height, channel_axis))

x = Convolution2D_bn(model_input, 32, 3, 3, strides=(2, 2), padding='valid')
x = Convolution2D_bn(x, 32, 3, 3, padding='valid')
x = Convolution2D_bn(x, 32, 3, 3)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = Convolution2D_bn(x, 40, 1, 1, padding='valid')
x = Convolution2D_bn(x, 48, 3, 3, padding='valid')
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

# mixed 0: 35 x 35 x 256
branch1x1 = Convolution2D_bn(x, 32, 1, 1)

branch5x5 = Convolution2D_bn(x, 48, 1, 1)
branch5x5 = Convolution2D_bn(branch5x5, 32, 5, 5)

branch3x3dbl = Convolution2D_bn(x, 32, 1, 1)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = Convolution2D_bn(branch_pool, 32, 1, 1)
x = concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    name='mixed0')

# mixed 1: 35 x 35 x 288
branch1x1 = Convolution2D_bn(x, 32, 1, 1)

branch5x5 = Convolution2D_bn(x, 48, 1, 1)
branch5x5 = Convolution2D_bn(branch5x5, 32, 5, 5)

branch3x3dbl = Convolution2D_bn(x, 32, 1, 1)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = Convolution2D_bn(branch_pool, 32, 1, 1)
x = concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    name='mixed1')

# mixed 2: 35 x 35 x 288
branch1x1 = Convolution2D_bn(x, 32, 1, 1)

branch5x5 = Convolution2D_bn(x, 48, 1, 1)
branch5x5 = Convolution2D_bn(branch5x5, 32, 5, 5)

branch3x3dbl = Convolution2D_bn(x, 32, 1, 1)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = Convolution2D_bn(branch_pool, 32, 1, 1)
x = concatenate(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],
    name='mixed2')

# mixed 3: 17 x 17 x 768
branch3x3 = Convolution2D_bn(x, 128, 3, 3, strides=(2, 2), padding='valid')
branch3x3dbl = Convolution2D_bn(x, 32, 1, 1)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3)
branch3x3dbl = Convolution2D_bn(branch3x3dbl, 48, 3, 3, strides=(2, 2), padding='valid')

branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        name='mixed3')

# mixed 4: 17 x 17 x 768
branch1x1 = Convolution2D_bn(x, 48, 1, 1)

branch7x7 = Convolution2D_bn(x, 64, 1, 1)
branch7x7 = Convolution2D_bn(branch7x7, 64, 1, 7)
branch7x7 = Convolution2D_bn(branch7x7, 48, 7, 1)

branch7x7dbl = Convolution2D_bn(x, 64, 1, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 64, 7, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 64, 1, 7)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 64, 7, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 1, 7)

branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
branch_pool = Convolution2D_bn(branch_pool, 48, 1, 1)
x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        name='mixed4')

# mixed 5, 6: 17 x 17 x 768
for i in range(2):
        branch1x1 = Convolution2D_bn(x, 48, 1, 1)

        branch7x7 = Convolution2D_bn(x, 48, 1, 1)
        branch7x7 = Convolution2D_bn(branch7x7, 48, 1, 7)
        branch7x7 = Convolution2D_bn(branch7x7, 48, 7, 1)

        branch7x7dbl = Convolution2D_bn(x, 48, 1, 1)
        branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 7, 1)
        branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 1, 7)
        branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 7, 1)
        branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Convolution2D_bn(branch_pool, 48, 1, 1)
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            name='mixed' + str(5 + i))

# mixed 7: 17 x 17 x 768
branch1x1 = Convolution2D_bn(x, 48, 1, 1)

branch7x7 = Convolution2D_bn(x, 48, 1, 1)
branch7x7 = Convolution2D_bn(branch7x7, 48, 1, 7)
branch7x7 = Convolution2D_bn(branch7x7, 48, 7, 1)

branch7x7dbl = Convolution2D_bn(x, 48, 1, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 7, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 1, 7)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 7, 1)
branch7x7dbl = Convolution2D_bn(branch7x7dbl, 48, 1, 7)

branch_pool = AveragePooling2D((3, 3),strides=(1, 1), padding='same')(x)
branch_pool = Convolution2D_bn(branch_pool, 48, 1, 1)
x = concatenate(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    name='mixed7')

# mixed 8: 8 x 8 x 1280
branch3x3 = Convolution2D_bn(x, 48, 1, 1)
branch3x3 = Convolution2D_bn(branch3x3, 128, 3, 3,strides=(2, 2), padding='valid')

branch7x7x3 = Convolution2D_bn(x, 48, 1, 1)
branch7x7x3 = Convolution2D_bn(branch7x7x3, 48, 1, 7)
branch7x7x3 = Convolution2D_bn(branch7x7x3, 48, 7, 1)
branch7x7x3 = Convolution2D_bn(branch7x7x3, 48, 3, 3, strides=(2, 2), padding='valid')

branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
x = concatenate(
    [branch3x3, branch7x7x3, branch_pool],
    name='mixed8')

    # mixed 9: 8 x 8 x 2048
for i in range(2):
        branch1x1 = Convolution2D_bn(x, 128, 1, 1)

        branch3x3 = Convolution2D_bn(x, 128, 1, 1)
        branch3x3_1 = Convolution2D_bn(branch3x3, 128, 1, 3)
        branch3x3_2 = Convolution2D_bn(branch3x3, 128, 3, 1)
        branch3x3 = concatenate(
            [branch3x3_1, branch3x3_2],
            name='mixed9_' + str(i))

        branch3x3dbl = Convolution2D_bn(x, 128, 1, 1)
        branch3x3dbl = Convolution2D_bn(branch3x3dbl, 128, 3, 3)
        branch3x3dbl_1 = Convolution2D_bn(branch3x3dbl, 128, 1, 3)
        branch3x3dbl_2 = Convolution2D_bn(branch3x3dbl, 128, 3, 1)
        branch3x3dbl = concatenate(
            [branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Convolution2D_bn(branch_pool, 48, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            name='mixed' + str(9 + i))

x = GlobalAveragePooling2D()(x)

x = Dense(128, kernel_regularizer=l2(1e-4), use_bias=True, name='Dense_1')(x)
x = Activation('relu', name='relu1')(x)
x = Dropout(DROPOUT)(x, training=True)

x = Dense(64, kernel_regularizer=l2(1e-4), use_bias=True, name='Dense_2')(x)
x = Activation('relu', name='relu2')(x)
x = Dropout(DROPOUT)(x, training=True)

x = Dense(num_classes, kernel_regularizer=l2(1e-4), use_bias=True, name='logits')(x)

model_output = Activation('softmax', name='Softmax')(x)
model = Model(model_input, model_output)
model.summary()

adam = Adam(lr=0.001,amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print('RGB')

from tensorflow.keras.callbacks import ModelCheckpoint, Callback, CSVLogger, ReduceLROnPlateau, EarlyStopping
class LossHistory(Callback):
      def on_train_begin(self, logs={}):
          self.losses = []

      def on_batch_end(self, batch, logs={}):
          self.losses.append(logs.get('loss'))
          
History = LossHistory()

path='D:/inception_v3_2/RGB/CNN/inception_v3_max.h5'
checkpointer1 = ModelCheckpoint(path, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=False)
checkpointer2 = ModelCheckpoint('D:/inception_v3_2/RGB/CNN/inception_model_{epoch:04d}.h5', period=2)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=0.000001)
es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)
csv_logger = CSVLogger('D:/inception_v3_2/RGB/CNN/inception_v3.csv', append=True, separator=';')


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
def generate_generator_multiple(generator,dir1, batch_size, img_height,img_width):
    dataX = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          batch_size = batch_size)
    while True:
            Xdata = dataX.next()
            yield Xdata[0], Xdata[1]  #Xdata[0] images and Xdata[1] labels

inputgenerator=generate_generator_multiple(generator=train_datagen,
                                           dir1=train_dir,
                                           batch_size=batch_size,
                                           img_height=img_height,
                                           img_width=img_height)       
     
validationgenerator=generate_generator_multiple(validation_datagen,
                                          dir1=validation_dir,
                                          batch_size=batch_size,
                                          img_height=img_height,
                                          img_width=img_height) 

print("fit_generator")
model.fit(inputgenerator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs = num_epochs,
                    validation_data = validationgenerator,
                    validation_steps = nb_validation_samples//batch_size,
                    callbacks=[History, checkpointer1, checkpointer2, csv_logger, es, reduce_lr],
                    shuffle=True,
                    verbose=1)

model.save('D:/inception_v3_2/RGB/CNN/inception_v3_final.h5')
model.save_weights('D:/inception_v3_2/RGB/CNN/inception_v3_weights_final.h5')
