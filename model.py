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
VGG_MEAN = [103.939, 116.779, 123.68]

class DeshadowNet:

    def __init__(self, x, deshadow_image, batch_size,keep_prob):
        
        self.batch_size = batch_size
        self.norm_x = norm_image(x)
        #self.norm_deshadow_image = norm_image(deshadow_image)
        
        self.A_input, self.S_input = self.G_Net(self.norm_x,keep_prob)
        #self.A_output = self.A_Net(self.norm_x,self.A_input,keep_prob)
        #self.S_output = self.S_Net(self.norm_x,self.S_input,keep_prob)
        self.A_output = self.A_Net(x,self.A_input,keep_prob)
        self.S_output = self.S_Net(x,self.S_input,keep_prob)

        self.shadow_matte= self.shadowMatte(self.A_output, self.S_output)
        self.gt_shadow_matte = self.calcShadowMatte(x, deshadow_image)

		# loss function
        self.loss = self.loss_function(self.shadow_matte, self.gt_shadow_matte)

        self.f_shadowfree = self.calcShadowFreeImage(x,tf.log(tf.clip_by_value(self.shadow_matte,1e-30,float('Inf'))))
        self.gt_shadowfree = self.calcShadowFreeImage(x,self.gt_shadow_matte)

        self.g_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_Net')
        self.a_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='A_Net')
        self.s_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='S_Net')

    def G_Net(self,x,keep_prob):
        vgg = vgg16.Vgg16()
        print('making g-network')
        with tf.variable_scope('G_Net'):
            #x = norm_image(x)
            A_input, S_input = vgg.build(x, self.batch_size,keep_prob)
        print('finishing g-network')
        return A_input, S_input

    def A_Net(self,x,G_input,keep_prob): # after conv3 in G_Net  256

        print('making a-network')
        sess=tf.Session()
        with tf.variable_scope('A_Net'):

            # conv2-1
            conv2_1 = conv_layer(x,[9,9,3,96],1,'conv2-1')

            # pool5
            conv2_1_output = max_pool_layer(conv2_1,[1,3,3,1],2,'pool2-1')
            print('conv2-1')
            print(sess.run(tf.shape(conv2_1_output)))

            # conv2-2
            conv2_2_output = conv_layer(G_input,[1,1,256,64],1,'conv2-2') 
            print('conv2-2')
            print(sess.run(tf.shape(conv2_2_output)))

            # concat conv2-1 and conv2-2
            conv_concat = tf.concat(axis=3, values = [conv2_1_output,conv2_2_output], name = 'concat_a_net')

            # conv2-3
            conv2_3 = conv_layer(conv_concat,[5,5,160,64],1,'conv2-3')
            print('conv2-3')
            print(sess.run(tf.shape(conv2_3)))

            # conv2-4
            conv2_4 = conv_layer(conv2_3,[5,5,64,64],1,'conv2-4')
            print('conv2-4')
            print(sess.run(tf.shape(conv2_4)))

            # conv2-5
            conv2_5 = conv_layer(conv2_4,[5,5,64,64],1,'conv2-5')
            print('conv2-5')
            print(sess.run(tf.shape(conv2_5)))

            # conv2-6
            conv2_6 = conv_layer(conv2_5,[5,5,64,64],1,'conv2-6')
            print('conv2-6')
            print(sess.run(tf.shape(conv2_6)))

            # deconv2_1
            deconv2_2 = deconv_layer(conv2_6,[4,4,3,64],[self.batch_size,224,224,3],2,'deconv2-2')
            print('deconv2-2')
            print(sess.run(tf.shape(deconv2_2)))
 
            print('finishing a-network')
            
            tf.summary.image('conv2_1',conv2_1_output[:,:,:,0:3])
            tf.summary.image('conv2_2',conv2_2_output[:,:,:,0:3])
            tf.summary.image('conv2_3',conv2_3[:,:,:,0:3])
            tf.summary.image('conv2_4',conv2_4[:,:,:,0:3])
            tf.summary.image('conv2_5',conv2_5[:,:,:,0:3])
            tf.summary.image('conv2_6',conv2_6[:,:,:,0:3])
            red = tf.reshape(deconv2_2[:,:,:,0], [-1,224,224,1])
            green = tf.reshape(deconv2_2[:,:,:,1], [-1,224,224,1])
            blue = tf.reshape(deconv2_2[:,:,:,2], [-1,224,224,1])
            tf.summary.image('deconv3_1',red)
            tf.summary.image('deconv3_1',green)
            tf.summary.image('deconv3_1',blue)
            #tf.summary.image('deconv2_2-1',deconv2_2[:,:,:,0:1])
            #tf.summary.image('deconv2_2-2',deconv2_2[:,:,:,1:2])
            #tf.summary.image('deconv2_2',deconv2_2[:,:,:,:])
            sess.close()
            return deconv2_2

    def S_Net(self,x,G_input,keep_prob): # after conv5 in G_Net 512
        
        print('making s-network')
        sess = tf.Session()
        with tf.variable_scope('S_Net'):
            # conv3-1
            conv3_1 = conv_layer(x,[9,9,3,96],1,'conv3-1')
            
            # pool5
            conv3_1_output = max_pool_layer(conv3_1,[1,3,3,1],2,'pool3-1')
            print('conv3-1')
            print(sess.run(tf.shape(conv3_1_output)))
            
            # conv2-2
            conv3_2_output = conv_layer(G_input,[1,1,256,64],1,'conv3-2') # need deconv before, also need to change the size of the channel
            print('conv3-2')
            print(sess.run(tf.shape(conv3_2_output)))

            # concat conv2-1 and conv2-2
            conv_concat = tf.concat(axis=3, values = [conv3_1_output,conv3_2_output],name = 'concat_s_net')

            # conv3-3
            conv3_3 = conv_layer(conv_concat,[5,5,160,64],1,'conv3-3')
            print('conv3-3')
            print(sess.run(tf.shape(conv3_3)))

            # conv3-4
            conv3_4 = conv_layer(conv3_3,[5,5,64,64],1,'conv3-4')
            print('conv3-4')
            print(sess.run(tf.shape(conv3_4)))

            # conv3-5
            conv3_5 = conv_layer(conv3_4,[5,5,64,64],1,'conv3-5')
            print('conv3-5')
            print(sess.run(tf.shape(conv3_5)))

            # conv3-6
            conv3_6 = conv_layer(conv3_5,[5,5,64,64],1,'conv3-6')
            print('conv3-6')
            print(sess.run(tf.shape(conv3_6)))

            # deconv3-2
            deconv3_2 = deconv_layer(conv3_6,[4,4,3,64],[self.batch_size,224,224,3],2,'decov3-2')
            print('decov3-2')
            print(sess.run(tf.shape(deconv3_2)))


            tf.summary.image('conv3_1',conv3_1_output[:,:,:,0:3])
            tf.summary.image('conv3_2',conv3_2_output[:,:,:,0:3])
            tf.summary.image('conv3_3',conv3_3[:,:,:,0:3])
            tf.summary.image('conv3_4',conv3_4[:,:,:,0:3])
            tf.summary.image('conv3_5',conv3_5[:,:,:,0:3])
            tf.summary.image('conv3_6',conv3_6[:,:,:,0:3])
            red = tf.reshape(deconv3_2[:,:,:,0], [-1,224,224,1])
            green = tf.reshape(deconv3_2[:,:,:,1], [-1,224,224,1])
            blue = tf.reshape(deconv3_2[:,:,:,2], [-1,224,224,1])
            tf.summary.image('deconv3_1',red)
            tf.summary.image('deconv3_1',green)
            tf.summary.image('deconv3_1',blue)
            #tf.summary.image('deconv3_2-1',deconv3_2[:,:,:,0:1])
            #tf.summary.image('deconv3_2-2',deconv3_2[:,:,:,1:2])
            #tf.summary.image('deconv3_2',deconv3_2[:,:,:,:])
            print('finishing s-network')
            sess.close()
            return deconv3_2

    def shadowMatte(self, A_input, S_input):
        with tf.variable_scope('F_Matte'):
            x = tf.concat([A_input,S_input],3,name='concat_a_s')
            x = conv_layer(x,[1,1,6,3],1,'1x1conv')
            #x  = tf.clip_by_value(x,0,1) 
            return x

    def loss_function(self, x, ground_truth):
        # MSE loss
        with tf.variable_scope('Loss'):
            #diff = tf.log(tf.clip_by_value(x,1e-10,255)) - tf.log(tf.clip_by_value(ground_truth,1e-10,255))
            diff = tf.log(tf.clip_by_value(x,1e-30,float('Inf'))) - ground_truth
            loss = tf.square(diff)
            output = tf.reduce_mean(loss)
        return output

    def calcShadowImage(self, deshadow, matte):
        with tf.variable_scope('Shadow_Img'):
            output = tf.multiply(deshadow,matte)
            return output

    def calcShadowFreeImage(self, shadow, matte):
        with tf.variable_scope('Shadow_Free_Img'):
            # log (I_s) = log (S_m) + log (I_ns)
            # log(I_ns) = log (I_s) - log (S_m) 
            #unnorm_shadow = unnorm_image(shadow)
            #matte_cliped = tf.clip_by_value(matte, 1e-10,255)
            shadow = tf.clip_by_value(shadow,1e-3,255)
            log_shadow = tf.log(shadow)
            log_shadowfree = log_shadow - matte
            #output = tf.divide(shadow,matte)
            #output = tf.clip_by_value(output, 0,255)
            shadowfree= tf.exp(log_shadowfree)  
            return tf.clip_by_value(shadowfree, 0,255)           
            #return unnorm_image(output)

    def calcShadowMatte(self, x, deshadow_image):
        with tf.variable_scope('GT_Matte'):
            # log (I_s) = log (S_m) + log (I_ns)
            # log (S_m) = log (I_s) - log(I_ns)
            # but in this case I_s/I_ns cannot be negative or zero
            # so what we cannot normalize the input images
            # then what happens to the output of network which is normalized?
            # matte should be between 0 to 1
            x = tf.clip_by_value(x,1e-3,255)
            deshadow_image = tf.clip_by_value(deshadow_image,1e-3,255)
            log_shadow_matte = tf.log(x) - tf.log(deshadow_image)
            log_shadow_matte = tf.clip_by_value(log_shadow_matte,-float('Inf'),tf.log(255.0))
            #return tf.exp(log_shadow_matte)
            return log_shadow_matte
def norm_image(x):
    with tf.name_scope('norm_vgg_input'):
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        return bgr

def unnorm_image(x):
    with tf.name_scope('unnorm_vgg_input'):
        # Convert BGR to RGB
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=x)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        rgb = tf.concat(axis=3, values=[
            red + VGG_MEAN[2],
            green + VGG_MEAN[1],
            blue + VGG_MEAN[0],
        ])
        assert rgb.get_shape().as_list()[1:] == [224, 224, 3]
        return rgb

def image_show(np_image):
    img = Image.fromarray(np_image,'RGB')
    img.show()