import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print(path)
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path

            print('path to the weight of vgg: {}'.format(path))

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print(self.data_dict)
        print("npy file loaded")

    def build(self, x, batch_size, keep_prob):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        """

        #with tf.name_scope('norm_vgg_input'):
        #    # Convert RGB to BGR
        #    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
        #    assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #    assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        #    bgr = tf.concat(axis=3, values=[
        #        blue - VGG_MEAN[0],
        #        green - VGG_MEAN[1],
        #        red - VGG_MEAN[2],
        #    ])
        #    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]


        sess=tf.Session()
        print('x')
        print(sess.run(tf.shape(x)))
        self.conv1_1 = self.conv_layer(x, keep_prob, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, keep_prob, "conv1_2")
        self.pool1 = self.max_pool_stride(self.conv1_2, 2,'pool1')
        print('conv1')
        print(sess.run(tf.shape(self.pool1)))

        self.conv2_1 = self.conv_layer(self.pool1, keep_prob, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, keep_prob, "conv2_2")
        self.pool2 = self.max_pool_stride(self.conv2_2, 2,'pool2')
        print('conv2')
        print(sess.run(tf.shape(self.pool2)))

        self.conv3_1 = self.conv_layer(self.pool2, keep_prob, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, keep_prob, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, keep_prob, "conv3_3")
        self.pool3 = self.max_pool_stride(self.conv3_3, 2,'pool3')
        print('conv3')
        print(sess.run(tf.shape(self.pool3)))
        self.deconv2_1= self.deconv_layer(self.pool3,[8,8,256,256],[batch_size,112,112,256],4,"deconv2_1")
        print('deconv2_1')
        print(sess.run(tf.shape(self.deconv2_1)))

        self.conv4_1 = self.conv_layer(self.pool3, keep_prob, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, keep_prob, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, keep_prob, "conv4_3")
        self.pool4 = self.max_pool_stride(self.conv4_3, 1, 'pool4')
        print('conv4')
        print(sess.run(tf.shape(self.pool4)))

        self.conv5_1 = self.conv_layer(self.pool4, keep_prob, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, keep_prob, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, keep_prob, "conv5_3")
        self.pool5 = self.max_pool_stride(self.conv5_3, 1, 'pool5')
        print('conv5')
        print(sess.run(tf.shape(self.pool5)))


       # fcn layer
        self.fcn_1 = self.fully_conv_layer(self.pool5, [1,1,512,4096], 1, keep_prob,'fcn1')
        print('fcn1')
        print(sess.run(tf.shape(self.fcn_1)))
        self.fcn_2 = self.fully_conv_layer(self.fcn_1, [1,1,4096,4096], 1, keep_prob, 'fcn2')
        print('fcn2')
        print(sess.run(tf.shape(self.fcn_2)))        
        self.fcn_3 = self.fully_conv_layer(self.fcn_2, [1,1,4096,1000], 1, keep_prob, 'fcn3')
        print('fcn2')
        print(sess.run(tf.shape(self.fcn_3)))           
        self.deconv3_1= self.deconv_layer(self.fcn_3 ,[8,8,256,1000],[batch_size,112,112,256],4,"deconv3_1")
        print('deconv3_1')
        print(sess.run(tf.shape(self.deconv3_1)))
        sess.close()

        self.data_dict = None

        tf.summary.image('conv1',self.pool1[:,:,:,0:3])
        tf.summary.image('conv2',self.pool2[:,:,:,0:3])
        tf.summary.image('conv3',self.pool3[:,:,:,0:3])
        tf.summary.image('conv4',self.pool4[:,:,:,0:3])
        tf.summary.image('conv5',self.pool5[:,:,:,0:3])
        tf.summary.image('conv5',self.pool5[:,:,:,0:3])
        red = tf.reshape(self.deconv2_1[:,:,:,0], [-1,112,112,1])
        green = tf.reshape(self.deconv2_1[:,:,:,1], [-1,112,112,1])
        blue = tf.reshape(self.deconv2_1[:,:,:,2], [-1,112,112,1])
        tf.summary.image('deconv2_1',red)
        tf.summary.image('deconv2_1',green)
        tf.summary.image('deconv2_1',blue)
        red = tf.reshape(self.deconv3_1[:,:,:,0], [-1,112,112,1])
        green = tf.reshape(self.deconv3_1[:,:,:,1], [-1,112,112,1])
        blue = tf.reshape(self.deconv3_1[:,:,:,2], [-1,112,112,1])
        tf.summary.image('deconv3_1',red)
        tf.summary.image('deconv3_1',green)
        tf.summary.image('deconv3_1',blue)
        return self.deconv2_1, self.deconv3_1

    def max_pool_stride(self, bottom, stide, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stide, stide, 1], padding='SAME', name=name)

    def fully_conv_layer(self, x, filtershape, stride, keep_prob, name):
        with tf.variable_scope(name):
            filters = tf.get_variable(
                name = 'weight',
                shape = filtershape,
                dtype = tf.float32,
			    initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
                trainable = True)
            conv = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')
            conv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
                                   trainable=True, name ='bias')
            bias = tf.nn.bias_add(conv, conv_biases)
            prelu = self.prelu(bias)
            #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
            return prelu

    def conv_layer(self, bottom, keep_prob, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            prelu = self.prelu(bias)
            #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
            return prelu

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")

    def prelu(self, x):
        with tf.variable_scope('prelu'):
            alphas = tf.get_variable('alpha', x.get_shape()[-1],
							initializer=tf.constant_initializer(0.0),
							dtype=tf.float32)
            pos = tf.nn.relu(x)
            neg = alphas * (x - abs(x)) * 0.5
            return pos + neg

    def deconv_layer(self, x, filtershape,output_shape, stride, name):
        with tf.variable_scope(name):
            filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
            deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')
            #deconv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
            #                       trainable=True, name ='bias')
            #bias = tf.nn.bias_add(deconv, deconv_biases)
            #prelu = self.prelu(bias)
            #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
            return self.prelu(deconv)
