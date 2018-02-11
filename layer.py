import tensorflow as tf

def conv_layer(x, filtershape, stride, name):
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
        output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('conv_filter',img_filt)
        return output

def deconv_layer(x, filtershape,output_shape, stride, name):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name = 'weight',
            shape = filtershape,
            dtype = tf.float32,
            initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
            trainable = True)
        deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')
        #deconv_biases = tf.Variable(tf.constant(0.0, shape = [filtershape[3]], dtype = tf.float32),
        #                        trainable=True, name ='bias')
        #bias = tf.nn.bias_add(deconv, deconv_biases)
        #output = prelu(bias)
        #output = tf.nn.dropout(prelu, keep_prob=keep_prob)
        img_filt = tf.reshape(filters[:,:,:,1], [-1,filtershape[0],filtershape[1],1])
        tf.summary.image('deconv_filter',img_filt)
        return prelu(deconv)

def max_pool_layer(x,filtershape,stride,name):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding ='SAME',name = name)

def prelu(x):
    with tf.variable_scope('prelu'):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5
        return pos + neg