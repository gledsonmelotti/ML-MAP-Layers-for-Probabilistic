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

import scipy.io as sio

print('Prediction')

test_dir = 'D:/RGB/Tres_classes_2/test'
num_classes = 3
img_height, img_width = 299, 299
model = load_model('D:/inception_v3_2/RGB/CNN/test/InceptionV3_RGB.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
def generate_generator_multiple_test(generator,dir1, batch_size, img_height,img_width):
    dataX = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          color_mode='rgb',
                                          class_mode = 'categorical',
                                          shuffle = False,
                                          batch_size = 1)
    while True:
            Xdata = dataX.next()
            yield Xdata[0]  #Xdata[0] images without labels
     
testgenerator=generate_generator_multiple_test(test_datagen,
                                          dir1=test_dir,
                                          img_height=img_height,
                                          img_width=img_height,
                                          batch_size=1)

from sklearn.metrics import precision_recall_fscore_support

test_Pedestrian_dir='D:/RGB/Tres_classes_2/test/Pedestrian'
testP_ids = next(os.walk(test_Pedestrian_dir))[2]

test_Car_dir='D:/RGB/Tres_classes_2/test/Car'
testC_ids = next(os.walk(test_Car_dir))[2]

test_Cyclist_dir='D:/RGB/Tres_classes_2/test/Cyclist'
testCy_ids = next(os.walk(test_Cyclist_dir))[2]

nb_test_samples = len(os.listdir(test_Pedestrian_dir)) + len(os.listdir(test_Car_dir)) + len(os.listdir(test_Cyclist_dir))
print('nb_test_samples')
print(nb_test_samples)

Label_Car_True = np.zeros((len(testC_ids),1),dtype=np.float32)
Label_Cyclist_True = np.ones((len(testCy_ids),1),dtype=np.float32)
Label_Pedestrian_True = 2*np.ones((len(testP_ids),1),dtype=np.float32)
Test_labels = np.concatenate((Label_Car_True,Label_Cyclist_True,Label_Pedestrian_True),axis=0)
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Test_labels.mat',{'Test_labels':Test_labels})

Probability = model.predict_generator(testgenerator,steps = nb_test_samples, verbose=1)
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Probability.mat',{'Probability':Probability})

Test_predict = np.argmax(Probability,axis=1)
sio.savemat('D:/inception_v3_2/RGB/CNN/test/Test_predict.mat',{'Test_predict':Test_predict})

precision, recall, fscore, support=precision_recall_fscore_support(Test_labels, Test_predict, average=None, labels=[0, 1, 2])
print('Os fscores de Cars, Cyclists e Pedestrians são')
print(fscore)
fscore_medio=np.sum(fscore)/3.
print(fscore_medio)