# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:45:36 2020

@author: Gledson
"""

import tensorflow as tf
import numpy as np
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
from tensorflow.keras import utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate 
from tensorflow.keras.layers import BatchNormalization, Convolution2D, Dense 
from tensorflow.keras.layers import Dropout, Flatten, Add, Lambda, GaussianDropout
from tensorflow.keras.layers import Average
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img

from tqdm import tqdm

import scipy.io as sio

# modalidade = 'RGB' ou 'RM' ou 'DM'

modalidades = ['RM', 'DM', 'RGB']
for m in range(len(modalidades)):
    modalidade = modalidades[m]
    
    if modalidade == 'DM':
        DirModalidade = 'BF_'+modalidade+'_13_13' 
        chanel_modalidade = 1
        color_modalidade = 'grayscale'
    elif modalidade == 'RM':
         DirModalidade = 'BF_'+modalidade+'_13_13' 
         chanel_modalidade = 1
         color_modalidade = 'grayscale'
    else:
         DirModalidade = 'RGB' 
         chanel_modalidade = 3
         color_modalidade = 'rgb'
    print()
    
    model = load_model('D:/inception_v3_2/'+modalidade+'/CNN/test/InceptionV3_'+modalidade+'.h5')
    model.summary()
    num_classes = 3
    img_height, img_width, channel = 299, 299, chanel_modalidade
    color_mode = color_modalidade
    interpolation = 'nearest'
    
    # To get the logits, we will have to remove the softmax layer from our model. 
    # Once the softmax model is removed, we can pass in the data as input and get 
    # the logits as output.
    model_before_softmax = Model(inputs=model.input, outputs=model.get_layer("logits").output)
    model_before_softmax.summary()
    
    print()
    print('D:/inception_v3_2/'+modalidade+'/CNN/test/InceptionV3_'+modalidade+'.h5')
    print()
    
    print('Logits: Saída antes da Softmax')
    print()
    
    print('##################### Validation Logits ######################')
    print()
    
    diretorio_Pedestrian='D:/'+DirModalidade+'/Tres_classes_2/validation/Pedestrian'
    diretorioP_ids = next(os.walk(diretorio_Pedestrian))[2]
    
    diretorio_Car='D:/'+DirModalidade+'/Tres_classes_2/validation/Car'
    diretorioC_ids = next(os.walk(diretorio_Car))[2]
    
    diretorio_Cyclist='D:/'+DirModalidade+'/Tres_classes_2/validation/Cyclist'
    diretorioCy_ids = next(os.walk(diretorio_Cyclist))[2]
    
    print('Predict and Probability to Pedestrian')
    probsP= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    PredsP=np.zeros(len(diretorioP_ids),dtype=np.float32)
    def predictP(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsP[n]= logits
            Auxprobs[:,0]=probsP[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsP[:,0] # Car in second column
            Auxprobs[:,2]=probsP[:,1] # Cyclist in last column
            probsP[n]=Auxprobs[n]
            PredsP[n]=np.argmax(probsP[n])   
    
    basedir = diretorio_Pedestrian
    test=diretorioP_ids
    predictP(basedir, model_before_softmax, test)
    
    print('Predict and Probability to Car')
    probsC= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    PredsC= np.zeros(len(diretorioC_ids),dtype=np.float32)
    def predictC(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsC[n]=logits
            Auxprobs[:,0]=probsC[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsC[:,0] # Car in second column
            Auxprobs[:,2]=probsC[:,1] # Cyclist in last column
            probsC[n]=Auxprobs[n]
            PredsC[n]=np.argmax(probsC[n])
    
    basedir = diretorio_Car
    test=diretorioC_ids
    predictC(basedir, model_before_softmax, test)
    
    print('Predict and Probability to Cyclist')
    probsCy= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    PredsCy= np.zeros(len(diretorioCy_ids),dtype=np.float32)
    def predictCy(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsCy[n]=logits
            Auxprobs[:,0]=probsCy[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsCy[:,0] # Car in second column
            Auxprobs[:,2]=probsCy[:,1] # Cyclist in last column
            probsCy[n]=Auxprobs[n]
            PredsCy[n]=np.argmax(probsCy[n])
            
    basedir = diretorio_Cyclist
    test=diretorioCy_ids
    predictCy(basedir, model_before_softmax, test)
    
    Val_logit_predict = np.concatenate((PredsP,PredsC,PredsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Val_logit_predict.mat',{'Val_logit_predict':Val_logit_predict})
    
    Probability_logits_val = np.concatenate((probsP,probsC,probsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Probability_logits_val.mat',{'Probability_logits_val':Probability_logits_val})
    
    Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
    Label_Car_True = np.ones((len(PredsC),1),dtype=np.float32)
    Label_Cyclist_True = 2.*np.ones((len(PredsCy),1),dtype=np.float32)
    Val_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True,Label_Cyclist_True),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Val_labels.mat',{'Val_labels':Val_labels})
    
    print()
    print('###################### Test Logits ###########################')
    print()
    
    diretorio_Pedestrian='D:/'+DirModalidade+'/Tres_classes_2/test/Pedestrian'
    diretorioP_ids = next(os.walk(diretorio_Pedestrian))[2]
    
    diretorio_Car='D:/'+DirModalidade+'/Tres_classes_2/test/Car'
    diretorioC_ids = next(os.walk(diretorio_Car))[2]
    
    diretorio_Cyclist='D:/'+DirModalidade+'/Tres_classes_2/test/Cyclist'
    diretorioCy_ids = next(os.walk(diretorio_Cyclist))[2]
    
    print('Predict and Probability to Pedestrian')
    probsP= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    PredsP=np.zeros(len(diretorioP_ids),dtype=np.float32)
    def predictP(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsP[n]=logits
            Auxprobs[:,0]=probsP[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsP[:,0] # Car in second column
            Auxprobs[:,2]=probsP[:,1] # Cyclist in last column
            probsP[n]=Auxprobs[n]
            PredsP[n]=np.argmax(probsP[n])   
    
    basedir = diretorio_Pedestrian
    test=diretorioP_ids
    predictP(basedir, model_before_softmax, test)
    
    print('Predict and Probability to Car')
    probsC= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    PredsC= np.zeros(len(diretorioC_ids),dtype=np.float32)
    def predictC(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsC[n]=logits
            Auxprobs[:,0]=probsC[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsC[:,0] # Car in second column
            Auxprobs[:,2]=probsC[:,1] # Cyclist in last column
            probsC[n]=Auxprobs[n]
            PredsC[n]=np.argmax(probsC[n])
    
    basedir = diretorio_Car
    test=diretorioC_ids
    predictC(basedir, model_before_softmax, test)
    
    print('Predict and Probability to Cyclist')
    probsCy= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    PredsCy= np.zeros(len(diretorioCy_ids),dtype=np.float32)
    def predictCy(basedir, model_before_softmax, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsCy[n]=logits
            Auxprobs[:,0]=probsCy[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsCy[:,0] # Car in second column
            Auxprobs[:,2]=probsCy[:,1] # Cyclist in last column
            probsCy[n]=Auxprobs[n]
            PredsCy[n]=np.argmax(probsCy[n])
         
    basedir = diretorio_Cyclist
    test=diretorioCy_ids
    predictCy(basedir, model_before_softmax, test)
    
    Test_logit_predict = np.concatenate((PredsP,PredsC,PredsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Test_logit_predict.mat',{'Test_logit_predict':Test_logit_predict})
    
    Probability_logits_test = np.concatenate((probsP,probsC,probsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Probability_logits_test.mat',{'Probability_logits_test':Probability_logits_test})
    
    Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
    Label_Car_True = np.ones((len(PredsC),1),dtype=np.float32)
    Label_Cyclist_True = 2.*np.ones((len(PredsCy),1),dtype=np.float32)
    Test_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True,Label_Cyclist_True),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Test_labels.mat',{'Test_labels':Test_labels})
    
    print()
    print('###################### Train Logits ###########################')
    print()
    
    diretorio_Pedestrian='D:/'+DirModalidade+'/Tres_classes_2/train/Pedestrian'
    diretorioP_ids = next(os.walk(diretorio_Pedestrian))[2]
    
    diretorio_Car='D:/'+DirModalidade+'/Tres_classes_2/train/Car'
    diretorioC_ids = next(os.walk(diretorio_Car))[2]
    
    diretorio_Cyclist='D:/'+DirModalidade+'/Tres_classes_2/train/Cyclist'
    diretorioCy_ids = next(os.walk(diretorio_Cyclist))[2]
    
    print('Predict and Probability to Pedestrian')
    probsP= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    PredsP=np.zeros(len(diretorioP_ids),dtype=np.float32)
    def predictP(basedir, model_before_softmax, train):
        for n, id_ in tqdm(enumerate(train), total=len(train)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsP[n]=logits
            Auxprobs[:,0]=probsP[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsP[:,0] # Car in second column
            Auxprobs[:,2]=probsP[:,1] # Cyclist in last column
            probsP[n]=Auxprobs[n]
            PredsP[n]=np.argmax(probsP[n])   
    
    basedir = diretorio_Pedestrian
    train=diretorioP_ids
    predictP(basedir, model_before_softmax, train)
    
    print('Predict and Probability to Car')
    probsC= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    PredsC= np.zeros(len(diretorioC_ids),dtype=np.float32)
    def predictC(basedir, model_before_softmax, train):
        for n, id_ in tqdm(enumerate(train), total=len(train)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsC[n]=logits
            Auxprobs[:,0]=probsC[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsC[:,0] # Car in second column
            Auxprobs[:,2]=probsC[:,1] # Cyclist in last column
            probsC[n]=Auxprobs[n]
            PredsC[n]=np.argmax(probsC[n])
    
    basedir = diretorio_Car
    train=diretorioC_ids
    predictC(basedir, model_before_softmax, train)
    
    print('Predict and Probability to Cyclist')
    probsCy= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    PredsCy= np.zeros(len(diretorioCy_ids),dtype=np.float32)
    def predictCy(basedir, model_before_softmax, train):
        for n, id_ in tqdm(enumerate(train), total=len(train)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model_before_softmax.predict(x)
            probsCy[n]=logits
            Auxprobs[:,0]=probsCy[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsCy[:,0] # Car in second column
            Auxprobs[:,2]=probsCy[:,1] # Cyclist in last column
            probsCy[n]=Auxprobs[n]
            PredsCy[n]=np.argmax(probsCy[n])
         
    basedir = diretorio_Cyclist
    train=diretorioCy_ids
    predictCy(basedir, model_before_softmax, train)
    
    Train_logit_predict = np.concatenate((PredsP,PredsC,PredsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Train_logit_predict.mat',{'Train_logit_predict':Train_logit_predict})
    
    Probability_logits_Train = np.concatenate((probsP,probsC,probsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Probability_logits_Train.mat',{'Probability_logits_Train':Probability_logits_Train})
    
    Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
    Label_Car_True = np.ones((len(PredsC),1),dtype=np.float32)
    Label_Cyclist_True = 2.*np.ones((len(PredsCy),1),dtype=np.float32)
    Train_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True,Label_Cyclist_True),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/Logits/Train_labels.mat',{'Train_labels':Train_labels})
    
    print()
    print('###################### Test Normal ###########################')
    print()
    
    diretorio_Pedestrian='D:/'+DirModalidade+'/Tres_classes_2/test/Pedestrian'
    diretorioP_ids = next(os.walk(diretorio_Pedestrian))[2]
    
    diretorio_Car='D:/'+DirModalidade+'/Tres_classes_2/test/Car'
    diretorioC_ids = next(os.walk(diretorio_Car))[2]
    
    diretorio_Cyclist='D:/'+DirModalidade+'/Tres_classes_2/test/Cyclist'
    diretorioCy_ids = next(os.walk(diretorio_Cyclist))[2]
    
    print('Predict and Probability to Pedestrian')
    probsP= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioP_ids),num_classes),dtype=np.float32)
    PredsP=np.zeros(len(diretorioP_ids),dtype=np.float32)
    def predictP(basedir, model, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model.predict(x)
            probsP[n]=logits
            Auxprobs[:,0]=probsP[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsP[:,0] # Car in second column
            Auxprobs[:,2]=probsP[:,1] # Cyclist in last column
            probsP[n]=Auxprobs[n]
            PredsP[n]=np.argmax(probsP[n])   
    
    basedir = diretorio_Pedestrian
    test=diretorioP_ids
    predictP(basedir, model, test)
    
    print('Predict and Probability to Car')
    probsC= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioC_ids),num_classes),dtype=np.float32)
    PredsC= np.zeros(len(diretorioC_ids),dtype=np.float32)
    def predictC(basedir, model, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model.predict(x)
            probsC[n]=logits
            Auxprobs[:,0]=probsC[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsC[:,0] # Car in second column
            Auxprobs[:,2]=probsC[:,1] # Cyclist in last column
            probsC[n]=Auxprobs[n]
            PredsC[n]=np.argmax(probsC[n])
    
    basedir = diretorio_Car
    test=diretorioC_ids
    predictC(basedir, model, test)
    
    print('Predict and Probability to Cyclist')
    probsCy= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    Auxprobs= np.zeros((len(diretorioCy_ids),num_classes),dtype=np.float32)
    PredsCy= np.zeros(len(diretorioCy_ids),dtype=np.float32)
    def predictCy(basedir, model, test):
        for n, id_ in tqdm(enumerate(test), total=len(test)):
            path = basedir + '/'+ id_
            img = load_img(path, color_mode = color_mode, target_size=(img_height, img_width), interpolation=interpolation)
            x = img_to_array(img)
            x= x / 255.
            x = np.expand_dims(x, axis=0)
            logits = model.predict(x)
            probsCy[n]=logits
            Auxprobs[:,0]=probsCy[:,2] # Pedestrian in first column
            Auxprobs[:,1]=probsCy[:,0] # Car in second column
            Auxprobs[:,2]=probsCy[:,1] # Cyclist in last column
            probsCy[n]=Auxprobs[n]
            PredsCy[n]=np.argmax(probsCy[n])
         
    basedir = diretorio_Cyclist
    test=diretorioCy_ids
    predictCy(basedir, model, test)
    
    Test_predict = np.concatenate((PredsP,PredsC,PredsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/test/Test_predict.mat',{'Test_predict':Test_predict})
    
    Probability = np.concatenate((probsP,probsC,probsCy),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/test/Probability.mat',{'Probability':Probability})
    sio.savemat('D:/Calibration/'+modalidade+'/Graficos_Calibrados_'+modalidade+'/Probability.mat',{'Probability':Probability})
    
    Label_Pedestrian_True = np.zeros((len(PredsP),1),dtype=np.float32)
    Label_Car_True = np.ones((len(PredsC),1),dtype=np.float32)
    Label_Cyclist_True = 2.*np.ones((len(PredsCy),1),dtype=np.float32)
    Test_labels = np.concatenate((Label_Pedestrian_True,Label_Car_True,Label_Cyclist_True),axis=0)
    sio.savemat('D:/Inception_v3_2/'+modalidade+'/CNN/test/Test_labels.mat',{'Test_labels':Test_labels})

