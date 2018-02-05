import tqdm
import cv2
import numpy as np
import tensorflow as tf
from model import DeshadowNet
from tfrecord_utils import *

from tensorflow.python import debug as tf_debug

IMAGE_SIZE = 224 # 64 128 224
BATCH_SIZE = 4
EPOCH = 500
G_LEARNING_RATE = 1e-5 #1e-5
LEARNING_RATE = 1e-2 #1e-4
MOMENTUM = 0.9

log_path = './graph/logs' # path to tensorboard graph
tfrecord_name = 'data.tfrecord'

# momentum to 0.9 and
# weight decay to 0.0005 (at this point we do not include)

'''
How to use Tensorboard
Run the command line: 
> tensorboard --logdir=./graph/logs
'''



def make_tfrecord():
    # make test set
    paths = glob.glob('./data/test/*')
    # making a list of image
    print('making pair of image list')
    image_list = []
    list_flag = True
    for path in paths:
        path = glob.glob(path+'/*')
        index = 0
        for sub in path:
            print('image name:{}'.format(sub))
            if list_flag:
                image_list.append([sub])
            else:
                image_list[index].append(sub)
                index += 1
        list_flag = False


    # writing tf record
    print('tf record writing start')
    tfrecord_filename = './data.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_filename)

    index = 0
    for img_pair in image_list:
        index += 1
        print('number: {}'.format(index))
        write_tfrecord_pair_img(writer, img_pair[0],img_pair[1],tfrecord_filename)

    writer.close()

    # testing tf record read1ng
    print('reading tf record')
    reader = tf.TFRecordReader()
    # closeshape, image = read_tfrecord(reader,tfrecord_filename)
    visualize = False
    read_tfrecord_pair_img(reader,tfrecord_filename,visualize)

def train(backupFlag):
    # setting random seed and reset graphs
    tf.set_random_seed(1111)
    tf.reset_default_graph()
    
    #calc step number
    step_num = int(408 / BATCH_SIZE)

    # place holders for shadow image input and shadow free image input
    with tf.variable_scope('Data_Input'):
        shadow = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], name ='Shadow_image')
        shadow_free = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3], name = 'Shadow_free_image')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # init network model
    model = DeshadowNet(shadow, shadow_free, BATCH_SIZE,keep_prob)

    # gpu options for some errors
    config = tf.ConfigProto()
    config.gpu_options.allocator_type ='BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.80

    with tf.Session() as sess:

        ## ==================== for tensorflow debugger ========================= ##
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        ## ====================================================================== ##

        print('initializing hyperparameter ... ')
        #global_step = tf.Variable(0, name='global_step', trainable=False) # step size
        epoch = tf.Variable(0, name='epoch', trainable=False) # epoch

        with tf.name_scope('Momentum_Opt'):
            print('init optimization')
            opt = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
            train_op = opt.minimize(model.loss)

        print('check if there is backup')
        if backupFlag and tf.train.get_checkpoint_state('./backup'):
            print('backup file loading ...')
            saver = tf.train.Saver()
            saver.restore(sess, './backup/latest')
        else:
            print('no backup ...')

        # loading dataset ...
        reader = tf.TFRecordReader()
        img_input, img_gt = read_tfrecord_pair_img(reader, tfrecord_name, False)

        coord = tf.train.Coordinator()

        # shuffle batch ...
        img_input_batch,img_gt_batch = tf.train.shuffle_batch([img_input, img_gt],
                                           BATCH_SIZE,
                                           capacity=10*BATCH_SIZE,
                                           min_after_dequeue = 2*BATCH_SIZE,
                                           num_threads=4,
                                           enqueue_many=False )

        print('initializing variables')
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        sess.run([init_op,init_local_op])
        threads = tf.train.start_queue_runners(coord = coord)
        
        tf.summary.scalar('epoch',epoch)
        summary_op = tf.summary.merge_all()
        print('start training ...')
        writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

        try:
            while not coord.should_stop():

                sess.run(tf.assign(epoch, tf.add(epoch,1)))
                print('epoch: {}'.format(sess.run(epoch)))
                if sess.run(epoch) <= EPOCH:
                    loss_val = 0
                    for i in tqdm.tqdm(range(step_num)):
                        # load image batched
                        img_input_seq, img_gt_seq = sess.run([img_input_batch,img_gt_batch])
                        img_input_seq.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
                        img_gt_seq.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])

                        # type casting
                        img_input_seq = img_input_seq.astype(np.float)
                        img_gt_seq = img_gt_seq.astype(np.float)

                        ## normalize
                        #img_input_seq = normalize_img(img_input_seq, BATCH_SIZE)
                        #img_gt_seq = normalize_img(img_gt_seq, BATCH_SIZE)

                        _,loss,summary = sess.run([train_op,model.loss,summary_op],
                                            feed_dict={
                                            shadow: img_input_seq,
                                            shadow_free:  img_gt_seq,
                                            keep_prob: 0.5})
                        # print('loss: {}'.format(loss))
                        loss_val += loss
                        # write log
                        writer.add_summary(summary, sess.run(epoch) * BATCH_SIZE + i)
                        

                        if i == floor(step_num/2) or i == (step_num -1):
                            A, S = sess.run([model.A_input,model.S_input],
                                    feed_dict={
                                    shadow: img_input_seq,
                                    shadow_free:  img_gt_seq,
                                    keep_prob: 1.0})
                            A.reshape([BATCH_SIZE,112,112,256])
                            S.reshape([BATCH_SIZE,112,112,256])                           
                            #image_show(A[0,:,:,0])
                            #image_show(A[0,:,:,0])                   
                            np.savetxt('./G_Net_Output/A{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*BATCH_SIZE+i)), A[0,:,:,0], delimiter=',')   # X is an array
                            np.savetxt('./G_Net_Output/S{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*BATCH_SIZE+i)), S[0,:,:,0], delimiter=',')   # X is an array
                    print('total loss in one epoch: {}'.format(loss_val))

                    # check for validation




                    # find shadow matte
                    f_shadow_val, gt_shadow_val = sess.run([model.f_shadow,model.gt_shadow],
                                            feed_dict={
                                            shadow: img_input_seq,
                                            shadow_free: img_gt_seq,
                                            keep_prob: 1.0})
                    # find
                    out = f_shadow_val[0].astype(np.uint8)
                    gt = gt_shadow_val[0].astype(np.uint8)
                    #image_show(out.astype(np.uint8))
                    #image_show(gt.astype(np.uint8))

                    #np.savetxt('f_shadow.out', out[:,:,0], delimiter=',')
                    #np.savetxt('gt_shadow.out', gt[:,:,0], delimiter=',')
                    cv2.imwrite('./output/f_deshadow/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                    cv2.imwrite('./output/gt_deshadow/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                    # saving data
                    saver = tf.train.Saver()
                    saver.save(sess, './backup/latest', write_meta_graph = False)
                    if sess.run(epoch) == EPOCH:
                        saver.save(sess,'./backup/fully_trained',write_meta_graph = False)

                    #summary = sess.run(summary_op)
            
                    

                else:
                    print('breaking out of while loop ...')
                    break

        except tf.errors.OutOfRangeError:
            print('ERROR: out of range\nDone Training -- epoch reached its limit')

        finally:
            print('FINAL: stop coordinate and joing thread')
            coord.request_stop()
            coord.join(threads)

def normalize_img(x, batch_size):

    norm_img_batch=[]
    for i in range(batch_size):
        norm_img = x[i]/ 255.0
        x[i] = norm_img
    return x

def unnorm_img(img):
    img = img * 255.0
    return img.astype(np.uint8)

def image_show(np_image):
    img = Image.fromarray(np_image,'RGB')
    img.show()

if __name__ == '__main__':
    # make_tfrecord()
    train(False)