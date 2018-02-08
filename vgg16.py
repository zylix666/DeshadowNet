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
        print("npy file loaded")
        # print(self.data_dict)

    def build(self, x, batch_size, keep_prob):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        # start_time = time.time()
        # print("build model started")

        # TODO
        # when making tf record and importing
        # consider the following structure below

        # rgb_scaled = rgb * 255.0
        #
        # Convert RGB to BGR
        #rgb_scaled = rgb * 255.0
        # unnormalize rgb image
        with tf.name_scope('unnorm_vgg_input'):
            #rgb_scaled = tf.scalar_mul(255.0,x)
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


        sess=tf.Session()
        print('x')
        print(sess.run(tf.shape(x)))
        self.conv1_1 = self.conv_layer(bgr, keep_prob, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, keep_prob, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        print('conv1')
        print(sess.run(tf.shape(self.pool1)))

        self.conv2_1 = self.conv_layer(self.pool1, keep_prob, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, keep_prob, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        print('conv2')
        print(sess.run(tf.shape(self.pool2)))

        self.conv3_1 = self.conv_layer(self.pool2, keep_prob, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, keep_prob, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, keep_prob, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        print('conv3')
        print(sess.run(tf.shape(self.pool3)))
        self.deconv2_1= self.deconv_layer(self.pool3,[8,8,256,256],[batch_size,112,112,256],4,"deconv2_1")
        print('deconv2_1')
        print(sess.run(tf.shape(self.deconv2_1)))

        self.conv4_1 = self.conv_layer(self.pool3, keep_prob, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, keep_prob, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, keep_prob, "conv4_3")
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        self.pool4 = self.max_pool_stride(self.conv4_3, 1, 'pool4')
        print('conv4')
        print(sess.run(tf.shape(self.pool4)))

        self.conv5_1 = self.conv_layer(self.pool4, keep_prob, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, keep_prob, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, keep_prob, "conv5_3")
        # self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        self.pool5 = self.max_pool_stride(self.conv5_3, 1, 'pool5')
        print('conv5')
        print(sess.run(tf.shape(self.pool5)))


       # fcn layer
        self.fcn_1 = self.fully_conv_layer(self.pool5, [1,1,512,4096], 1,'fcn1')
        print('fcn1')
        print(sess.run(tf.shape(self.fcn_1)))
        self.fcn_2 = self.fully_conv_layer(self.fcn_1, [1,1,4096,4096], 1, 'fcn2')
        print('fcn2')
        print(sess.run(tf.shape(self.fcn_2)))        
       
        #self.deconv3_1= self.deconv_layer(self.pool5 ,[8,8,256,512],[batch_size,112,112,256],4,"deconv3_1")
        self.deconv3_1= self.deconv_layer(self.fcn_2 ,[8,8,256,4096],[batch_size,112,112,256],4,"deconv3_1")
        print('deconv3_1')
        print(sess.run(tf.shape(self.deconv3_1)))
        sess.close()

        self.data_dict = None
        # print(("build model finished: %ds" % (time.time() - start_time)))
        return self.deconv2_1, self.deconv3_1

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_stride(self, bottom, stide, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stide, stide, 1], padding='SAME', name=name)

    def fully_conv_layer(self, x, filtershape, stride,name):
        with tf.variable_scope(name):
            filters = tf.get_variable(
                name = 'weight',
                shape = filtershape,
                dtype = tf.float32,
			    initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
                trainable = True)
            output = tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')
            return self.prelu(output)

    def conv_layer(self, bottom, keep_prob, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            prelu = self.prelu(bias)
            # relu = tf.nn.relu(bias)
            output = tf.nn.dropout(prelu, keep_prob=keep_prob)
            return output

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

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
            initializer = tf.contrib.layers.xavier_initializer(),
            trainable = True)
            deconv_output= tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')
            return self.prelu(deconv_output)
