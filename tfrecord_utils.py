'''
tfrecord utils

edited version from standford tensorflow class
'''

import os
import sys


import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

IMAGE_SIZE = 224

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    # image = Image.open(filename)
    # image = np.asarray(image, np.uint8)
    # shape = np.array(image.shape, np.int32)

    image = cv.imread(filename)
    image = cv.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # cv.imshow('test',image)
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # cv.waitKey()
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tostring() #image.tobytes() # convert image to raw data bytes in the array.

def write_to_tfrecord_pair_img(writer, binary_image1, binary_image2, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
                'image_input': _bytes_feature(binary_image1),
                'image_gt': _bytes_feature(binary_image2)
                }))
    writer.write(example.SerializeToString())

def write_tfrecord_pair_img(writer, image_file1, image_file2, tfrecord_file):
    shape1 ,binary_image1 = get_image_binary(image_file1)
    shape2 ,binary_image2 = get_image_binary(image_file2)
    write_to_tfrecord_pair_img(writer, binary_image1, binary_image2, tfrecord_file)

def read_from_tfrecord_pair_img(reader, filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'image_input': tf.FixedLenFeature([], tf.string),
                            'image_gt': tf.FixedLenFeature([], tf.string),
                        }, name='features')

    # image was saved as uint8, so we have to decode as uint8.
    img_input = tf.decode_raw(tfrecord_features['image_input'], tf.uint8)
    img_gt    = tf.decode_raw(tfrecord_features['image_gt'],    tf.uint8)

    # the image tensor is flattened out, so we have to reconstruct the shape
    img_input = tf.reshape(img_input, [IMAGE_SIZE, IMAGE_SIZE,3])
    img_gt = tf.reshape(img_gt, [IMAGE_SIZE, IMAGE_SIZE,3])
    return img_input, img_gt

def read_tfrecord_pair_img(reader, tfrecord_file,vis):
    img_input, img_gt = read_from_tfrecord_pair_img(reader, [tfrecord_file])
    if vis:
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(4):
                shadow, deshadow = sess.run([img_input,img_gt])
                pilimg = Image.fromarray(shadow)
                pilimg.show()
                pilimg = Image.fromarray(deshadow)
                pilimg.show()
            coord.request_stop()
            coord.join(threads)
            sess.close()

    return img_input, img_gt

def main():
    paths = glob.glob('./data/*')
    tfrecord_filename = './data.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)
    print('tf record writing start')

    image_size = 128;
    for path in paths:
        path = path + '/*'
        path = glob.glob(path)
        for subpath in path:
            x=[]
            if 'curry' in subpath:
                print('image name: {}'.format(subpath))
                # write_tfrecord(writer, subpath,tfrecord_filename)
                write_tfrecord(writer, subpath,tfrecord_filename)

    writer.close()

    # testing tf record read1ng
    print('reading tf record')
    reader = tf.TFRecordReader()
    tfrecord_filename = './test.tfrecord'

    # closeshape, image = read_tfrecord(reader,tfrecord_filename)
    image, shape = read_vis_tfrecord(reader,tfrecord_filename)
    print('shape:{}:'.format(shape))

if __name__ == '__main__':
    main()
