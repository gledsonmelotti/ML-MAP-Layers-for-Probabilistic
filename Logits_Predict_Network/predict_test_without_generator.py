# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:08:44 2020

@author: Gledson
"""


import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
#tf.get_logger().setLevel("WARNING")
#tf.autograph.set_verbosity(2)
tf.get_logger().setLevel('ERROR')
print(tf.__version__)

#import keras
#print("keras      {}".format(keras.__version__))
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

import scipy.io as sio

print('Prediction')

test_dir='D:/RGB/Tres_classes_2/test'
test_Pedestrian_dir='D:/RGB/Tres_classes_2/test/Pedestrian'
test_Car_dir='D:/RGB/Tres_classes_2/test/Car'
test_Cyclist_dir='D:/RGB/Tres_classes_2/test/Cyclist'

###############################################################################
nb_test_samples = len(os.listdir(test_Pedestrian_dir)) + len(os.listdir(test_Car_dir)) + len(os.listdir(test_Cyclist_dir))
print('######################################################################')
print('nb_test_samples')
print(nb_test_samples)
print('######################################################################')

num_classes = 3
color_mode = 'rgb'
interpolation='nearest'
img_width, img_height, channel_axis = 299, 299, 3

from keras_preprocessing.image.utils import load_img
from sklearn.metrics import precision_recall_fscore_support

model = load_model('D:/inception_v3_2/RGB/CNN/test/InceptionV3_RGB.h5')

test_Pedestrian_dir_RGB='D:/RGB/Tres_classes_2/test/Pedestrian'
testP_ids = next(os.walk(test_Pedestrian_dir_RGB))[2]

test_Car_dir_RGB='D:/RGB/Tres_classes_2/test/Car'
testC_ids = next(os.walk(test_Car_dir_RGB))[2]

test_Cyclist_dir_RGB='D:/RGB/Tres_classes_2/test/Cyclist'
testCy_ids = next(os.walk(test_Cyclist_dir_RGB))[2]

print('Predict and Probability to Pedestrian')
probsP= np.zeros((len(testP_ids),num_classes),dtype=np.float32)
Auxprobs= np.zeros((len(testP_ids),num_classes),dtype=np.float32)
PredsP=np.zeros(len(testP_ids),dtype=np.float32)
def predictP(basedir, model, test):
    for n, id_ in tqdm(enumerate(test), total=len(test)):
        path = basedir + '/'+ id_
        img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
        x = img_to_array(img)
        x= x / 255.
        x = np.expand_dims(x, axis=0)
        probsP[n]=model.predict(x)
        Auxprobs[:,0]=probsP[:,2] # Pedestrian in first column
        Auxprobs[:,1]=probsP[:,0] # Car in second column
        Auxprobs[:,2]=probsP[:,1] # Cyclist in last column
        probsP[n]=Auxprobs[n]
        PredsP[n]=np.argmax(probsP[n])   

basedir = test_Pedestrian_dir_RGB
test=testP_ids
predictP(basedir, model, test)

print('Predict and Probability to Car')
probsC= np.zeros((len(testC_ids),num_classes),dtype=np.float32)
Auxprobs= np.zeros((len(testC_ids),num_classes),dtype=np.float32)
PredsC= np.zeros(len(testC_ids),dtype=np.float32)
def predictC(basedir, model, test):
    for n, id_ in tqdm(enumerate(test), total=len(test)):
        path = basedir + '/'+ id_
        img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
        x = img_to_array(img)
        x= x / 255.
        x = np.expand_dims(x, axis=0)
        probsC[n]=model.predict(x)
        Auxprobs[:,0]=probsC[:,2] # Pedestrian in first column
        Auxprobs[:,1]=probsC[:,0] # Car in second column
        Auxprobs[:,2]=probsC[:,1] # Cyclist in last column
        probsC[n]=Auxprobs[n]
        PredsC[n]=np.argmax(probsC[n])

basedir = test_Car_dir_RGB
test=testC_ids
predictC(basedir, model, test)

print('Predict and Probability to Cyclist')
probsCy= np.zeros((len(testCy_ids),num_classes),dtype=np.float32)
Auxprobs= np.zeros((len(testCy_ids),num_classes),dtype=np.float32)
PredsCy= np.zeros(len(testCy_ids),dtype=np.float32)
def predictCy(basedir, model, test):
    for n, id_ in tqdm(enumerate(test), total=len(test)):
        path = basedir + '/'+ id_
        img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
        x = img_to_array(img)
        x= x / 255.
        x = np.expand_dims(x, axis=0)
        probsCy[n]=model.predict(x)
        Auxprobs[:,0]=probsCy[:,2] # Pedestrian in first column
        Auxprobs[:,1]=probsCy[:,0] # Car in second column
        Auxprobs[:,2]=probsCy[:,1] # Cyclist in last column
        probsCy[n]=Auxprobs[n]
        PredsCy[n]=np.argmax(probsCy[n])
        
basedir = test_Cyclist_dir_RGB
test=testCy_ids
predictCy(basedir, model, test)

########################## Unindo Pedestres Carros e Ciclistas ###################
Test_predict = np.concatenate((PredsP,PredsC,PredsCy),axis=0)
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Test_predict.mat',{'Test_predict':Test_predict})

Probability = np.concatenate((probsP,probsC,probsCy),axis=0)
Probability_test = Probability
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Probability.mat',{'Probability':Probability})

Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
Label_Car_True = np.ones((len(PredsC),1),dtype=np.float32)
Label_Cyclist_True = 2.*np.ones((len(PredsCy),1),dtype=np.float32)
Test_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True,Label_Cyclist_True),axis=0)
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Test_labels.mat',{'Test_labels':Test_labels})

precision, recall, fscore, support=precision_recall_fscore_support(Test_labels, Test_predict, average=None, labels=[0, 1, 2])

print('########################################################################')
print('Os fscores de Pedestrians, Cars e Cyclists são')
print(fscore)
print('########################################################################')
print('O fscore médio é:')
print(np.sum(fscore)/3.)
print('########################################################################')
