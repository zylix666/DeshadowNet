import tensorflow as tf

def conv_layer(x, filtershape, stride):
    filters = tf.get_variable(
        name = 'weight',
        shape = filtershape,
        dtype = tf.float32,
        initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
        #initializer = tf.contrib.layers.xavier_initializer(),
        trainable = True)
    return tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding= 'SAME')

def deconv_layer(x, filtershape,output_shape, stride):
    filters = tf.get_variable(
        name = 'weight',
        shape = filtershape,
        dtype = tf.float32,
        initializer = tf.random_normal_initializer(mean=0,stddev=0.001),
        #initializer = tf.contrib.layers.xavier_initializer(),
        trainable = True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding ='SAME')

def max_pool_layer(x,filtershape,stride):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding ='SAME')

def prelu(x):
  alphas = tf.get_variable('alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(x)
  neg = alphas * (x - tf.abs(x)) * 0.5

  return pos + neg
