from layer import *
import numpy as np
import math
from PIL import Image
import vgg16
import utils_vgg

from tensorflow.python import debug as tf_debug

##import vgg using keras
##from keras.applications.vgg16 import VGG16
##from keras.preprocessing import image
##from keras.applications.vgg16 import preprocess_input
##from keras import backend as K
#from tensorflow.python.keras.models import Model, Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, Dropout
#from tensorflow.python.keras.applications import VGG16
#from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.optimizers import Adam, RMSprop

## import vgg using slim
#import tensorflow.contrib.slim as slim
#import tensorflow.contrib.slim.nets


'''
Tensorflow + Keras Example
    https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

Keras Finetune , transfer learning
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb
    https://keras.io/applications/

Tensorflow
    https://github.com/machrisaa/tensorflow-vgg
'''

#TODO
#1. choose one of the following for fine tuning
#    - use tf slim for loading model
#    - use keras and tf for loading model
#2. data generate
#3. check if the code works using training
#4. data augmentation
#    - use keras if in 1 works with keras
#5. data generation
#    - use python script to generate data within maya env

MOMENTUM = 0.9

class DeshadowNet:
    def __init__(self, x, deshadow_image, batch_size,keep_prob):
        self.batch_size = batch_size
        self.A_input, self.S_input = self.G_Net(x,keep_prob)
        self.S_output = self.S_Net(x,self.S_input,keep_prob)
        self.A_output = self.A_Net(x,self.A_input,keep_prob)

        self.shadow_matte= self.shadowMatte(self.A_output, self.S_output)
        self.gt_shadow_matte = self.calcShadowMatte(x,deshadow_image)

        self.f_shadow = self.calcShadowImage(self.shadow_matte,deshadow_image)
        self.gt_shadow = self.calcShadowImage(self.gt_shadow_matte,deshadow_image)

        #calcShadowFreeImage(self, shadow, matte)
        self.f_shadowfree = self.calcShadowFreeImage(x,self.shadow_matte)
        self.gt_shadowfree = self.calcShadowFreeImage(x,self.gt_shadow_matte)

        self.loss = self.loss_function(self.shadow_matte, self.gt_shadow_matte)
        


        self.g_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Net')
        self.a_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='A_Net')
        self.s_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S_Net')




    def G_Net(self,x,keep_prob):
        vgg = vgg16.Vgg16()
        print('making g-network')
        with tf.variable_scope('G_Net'):
            A_input, S_input = vgg.build(x, self.batch_size,keep_prob)
        print('finishing g-network')
        return A_input, S_input

    def A_Net(self,x,G_input,keep_prob): # after conv3 in G_Net  256
        #x = rgb2bgr(x)
        print('making a-network')
        sess=tf.Session()
        with tf.variable_scope('A_Net'):
            # conv2-1
            with tf.variable_scope('conv2-1'):
                x = conv_layer(x,[9,9,3,96],1)
                bias  = tf.Variable(tf.constant(0.0, shape = [96], dtype = tf.float32),
                                   trainable=True, name ='conv64-1_bias')
                x  = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            # pool5
            with tf.variable_scope('pool2-1'):
                conv2_1_output = max_pool_layer(x,[1,3,3,1],2)

            print('conv2-1')
            print(sess.run(tf.shape(conv2_1_output)))

            # conv2-2
            with tf.variable_scope('conv2-2'):
                x = conv_layer(G_input,[1,1,256,64],1) # need deconv before, also need to change the size of the channel
                bias  = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable=True, name ='conv64-1_bias')
                x  = tf.nn.bias_add(x,bias)
                conv2_2_output = prelu(x)
                conv2_2_output = tf.nn.dropout(conv2_2_output, keep_prob=keep_prob)
            print('conv2-2')
            print(sess.run(tf.shape(conv2_2_output)))

            # concat conv2-1 and conv2-2
            with tf.variable_scope('concat_a_net'):
                x = tf.concat(axis=3, values = [conv2_1_output,conv2_2_output])

            # conv2-3
            with tf.variable_scope('conv2-3'):
                x = conv_layer(x,[5,5,160,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv2-3')
            print(sess.run(tf.shape(x)))
            # conv2-4
            with tf.variable_scope('conv2-4'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv2-4')
            print(sess.run(tf.shape(x)))
            # conv2-5
            with tf.variable_scope('conv2-5'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('pool2-5')
            print(sess.run(tf.shape(x)))
            # conv2-6
            with tf.variable_scope('conv2-6'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('pool2-6')
            print(sess.run(tf.shape(x)))

            # deconv
            with tf.variable_scope('deconv2-2'):
                x = deconv_layer(x,[4,4,3,64],[self.batch_size,224,224,3],2)
                x = prelu(x)
            print('finishing a-network')
            print('deconv2-1')
            print(sess.run(tf.shape(x)))
        sess.close()
        return x

    def S_Net(self,x,G_input,keep_prob): # after conv5 in G_Net 512
        #x = rgb2bgr(x)
        print('making s-network')
        sess = tf.Session()
        with tf.variable_scope('S_Net'):
            # conv2-1
            with tf.variable_scope('conv3-1'):
                x = conv_layer(x,[9,9,3,96],1)
                bias  = tf.Variable(tf.constant(0.0, shape = [96], dtype = tf.float32),
                                   trainable=True, name ='conv64-1_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            # pool5
            with tf.variable_scope('pool3-1'):
                conv3_1_output = max_pool_layer(x,[1,3,3,1],2)
            print('conv3-1')
            print(sess.run(tf.shape(conv3_1_output)))
            # conv2-2
            with tf.variable_scope('conv3-2'):
                x = conv_layer(G_input,[1,1,256,64],1) # need deconv before, also need to change the size of the channel
                bias  = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable=True, name ='conv64-1_bias')
                x = tf.nn.bias_add(x,bias)
                conv3_2_output = prelu(x)
                conv3_2_output = tf.nn.dropout(conv3_2_output, keep_prob=keep_prob)
            print('conv3-2')
            print(sess.run(tf.shape(conv3_2_output)))

            # concat conv2-1 and conv2-2
            with tf.variable_scope('concat_a_net'):
                x = tf.concat(axis=3, values = [conv3_1_output,conv3_2_output])

            # conv2-3
            with tf.variable_scope('conv3-3'):
                x = conv_layer(x,[5,5,160,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv3-3')
            print(sess.run(tf.shape(x)))
            # conv2-4
            with tf.variable_scope('conv3-4'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv3-4')
            print(sess.run(tf.shape(x)))
            # conv2-5
            with tf.variable_scope('conv3-5'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv3-5')
            print(sess.run(tf.shape(x)))
            # conv2-6
            with tf.variable_scope('conv3-6'):
                x = conv_layer(x,[5,5,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x = prelu(x)
                x = tf.nn.dropout(x, keep_prob=keep_prob)
            print('conv3-6')
            print(sess.run(tf.shape(x)))

            # deconv
            with tf.variable_scope('decov3-2'):
                x = deconv_layer(x,[4,4,3,64],[self.batch_size,224,224,3],2)
                x = prelu(x)
            print('decov3-2')
            print(sess.run(tf.shape(x)))
        print('finishing s-network')
        sess.close()
        return x

    def loss_function(self, x, ground_truth):
        # diff = np.log(x, out=np.zeros_like(x), where=x!=0) - np.log(ground_truth, out=np.zeros_like(ground_truth), where=ground_truth!=0)
        # diff = tf.log(tf.clip_by_value(x,1e-10,255) - np.log(ground_truth, out=np.zeros_like(ground_truth), where=ground_truth!=0)
        with tf.variable_scope('Loss'):
            diff = tf.log(tf.clip_by_value(x,1e-10,255)) - tf.log(tf.clip_by_value(ground_truth,1e-10,255))
            loss = tf.square(diff)
            output = tf.reduce_mean(loss)
        return output

    def calcShadowImage(self, deshadow, matte):
        with tf.variable_scope('Shadow_Img'):
            output = tf.multiply(deshadow,matte)
        return output

    def calcShadowFreeImage(self, shadow, matte):
        with tf.variable_scope('Shadow_Free_Img'):
            matte_cliped = tf.clip_by_value(matte, 1e-10,255)
            output = tf.divide(shadow,matte_cliped)        
            return output

    def shadowMatte(self, A_input, S_input):
        with tf.variable_scope('F_Matte'):
            x = tf.concat([A_input,S_input],3,name='concat_a_s')
            x = conv_layer(x,[1,1,6,3],1)
            #sess = tf.Session()
            #print('A shape')
            #print(sess.run(tf.shape(A_input)))
            #print('S shape')
            #print(sess.run(tf.shape(S_input)))
            #print('matt shape')
            #print(sess.run(tf.shape(x)))
            #sess.close()
        return x

    def calcShadowMatte(self, x, deshadow_image):
        with tf.variable_scope('GT_Matte'):
            # with tf.variable_scope('calc_shadow_matte'):
            deshadow_value_cliped = tf.clip_by_value(deshadow_image,1e-10,255)
            x_value_cliped = tf.clip_by_value(x,1e-10,255)
            output = tf.divide(x_value_cliped,deshadow_value_cliped)
            # output = np.divide(x, deshadow_image, out=np.zeros_like(x), where=deshadow_image!=0)
        return output


def image_show(np_image):
    img = Image.fromarray(np_image,'RGB')
    img.show()

def rgb2bgr(x):
    with tf.name_scope('rgb2bgr'):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue,
            green,
            red,
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
    return bgr

# not using fc layers
if False:
    '''
    fc layer for the VGG network
    '''
    # fc1
    with tf.variable_scopr('fc4096-1'):
        shape  = int( np.prod(x.get_shape()[1:] ) )
        weight = tf.Variable(tf.truncated_normal([shape,4096],
                                dtype = tf.float32,
                                stddev=1e-1), name= 'fc_weight')
        bias = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable=True, name='fc_bias')

        x = tf.reshape(x,[-1,shape])

        x = tf.nn.bias_add(tf.matmul(x,weight),bias)

    # fc2
    with tf.variable_scopr('fc4096-2'):
        weight = tf.Variable(tf.truncated_normal([4096,4096],
                                dtype = tf.float32,
                                stddev=1e-1), name= 'fc_weight')
        bias = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable=True, name='fc_bias')
        x = tf.nn.bias_add(tf.matmul(x,weight),bias)

    # fc3
    with tf.variable_scopr('fc1000-1'):
        weight = tf.Variable(tf.truncated_normal([4096,1000],
                                dtype = tf.float32,
                                stddev=1e-1), name= 'fc_weight')
        bias = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                trainable=True, name='fc_bias')
        x = tf.nn.bias_add(tf.matmul(x,weight),bias)
if False:
    def G_Net_keras(self,x):
        #K.set_session(sess)

        model = VGG16(weights='imagenet', include_top=False)
        A_input = model.get_layer('block3_pool ') # (None, 28, 28, 256)
        S_input = model.get_layer('block5_pool') # (None, 7, 7, 512)
        # printing out layer shape and type
        A_input.output
        S_input.output

        # making new model
        new_model = Model(input=model.input, outputs = [A_input.output,S_input.output])

        self.A_input = A_input.output
        self.S_input = S_input.output

        # deconv
        # deconv2-1
        with tf.variable_scope('decov2-1'):
            self.A_input = deconv_layer(A_input.output,[8,8,512,256],[self.batch_size,112,112,3],4)
        A_input2 = x
        # deconv3-1
        with tf.variable_scope('decov3-1'):
            self.S_input = deconv_layer(S_input.output,[8,8,3,256],[self.batch_size,112,112,3],4)
        S_input2 = x

        return self.A_input, self.S_input

    def G_Net(self, x):
        # VGG16 network
        with tf.variable_scope('g_network'):
            # conv1
            with tf.variable_scope('conv64-1'):
                x = conv_layer(x,[3,3,3,64],1)
                bias  = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable=True, name ='conv64-1_bias')
                x  = tf.nn.bias_add(x,bias)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv64-2'):
                x = conv_layer(x,[3,3,64,64],1)
                bias = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
                                   trainable = True, name = 'conv64-2_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            # pool1
            with tf.variable_scope('pool-1'):
                x = max_pool_layer(x,[1,2,2,1],2)
            # conv2
            with tf.variable_scope('conv128-1'):
                x = conv_layer(x,[3,3,64,128],1)
                bias = tf.Variable(tf.constant(0.0, shape = [129], dtype = tf.float32),
                                   trainable = True, name = 'conv128-1_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv128-2'):
                x = conv_layer(x,[3,3,128,128],1)
                bias = tf.Variable(tf.constant(0.0, shape = [128], dtype = tf.float32),
                                   trainable = True, name = 'conv128-2_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            # pool2
            with tf.variable_scope('pool-2'):
                x = max_pool_layer(x,[1,2,2,1],2)
            # conv3
            with tf.variable_scope('conv256-1'):
                x = conv_layer(x,[3,3,128,256],1)
                bias = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32),
                                   trainable = True, name = 'conv256-1_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv256-2'):
                x = conv_layer(x,[3,3,256,256],1)
                bias = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32),
                                   trainable = True, name = 'conv256-2_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv256-3'):
                x = conv_layer(x,[3,3,256,256],1)
                bias = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32),
                                   trainable = True, name = 'conv256-3_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            # pool3
            with tf.variable_scope('pool-3'):
                x = max_pool_layer(x,[1,2,2,1],2)
            # deconv2-1
            with tf.variable_scope('decov2-1'):
                x = deconv_layer(x,[8,8,512,256],[self.batch_size,112,112,3],4)
            A_input2 = x
            # conv4
            with tf.variable_scope('conv512-1'):
                x = conv_layer(x,[3,3,256,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-1_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv512-2'):
                x = conv_layer(x,[3,3,512,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-2_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv512-3'):
                x = conv_layer(x,[3,3,512,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-3_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            # pool4
            with tf.variable_scope('pool-1'):
                x = max_pool_layer(x,[1,2,2,1],2)
            # conv5
            with tf.variable_scope('conv512-4'):
                x = conv_layer(x,[3,3,512,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-4_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv512-5'):
                x = conv_layer(x,[3,3,512,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-5_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            with tf.variable_scope('conv512-6'):
                x = conv_layer(x,[3,3,512,512],1)
                bias = tf.Variable(tf.constant(0.0, shape = [512], dtype = tf.float32),
                                   trainable = True, name = 'conv512-6_bias')
                x = tf.nn.bias_add(x,bias)
                x  =tf.nn.relu(x)
            # pool5
            with tf.variable_scope('pool-1'):
                x = max_pool_layer(x,[1,2,2,1],2)
            # deconv3-1
            with tf.variable_scope('decov3-1'):
                x = deconv_layer(x,[8,8,3,256],[self.batch_size,112,112,3],4)
            S_input2 = x
        return A_input2, S_input2
