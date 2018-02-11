import tqdm
import cv2
import numpy as np
import tensorflow as tf
from model import DeshadowNet
from tfrecord_utils import *

from tensorflow.python import debug as tf_debug

IMAGE_SIZE = 224 # 64 128 224
BATCH_SIZE = 2
EPOCH = 50000000
INIT_G_LEARNING_RATE = 1e-5 #1E-5
INIT_LEARNING_RATE = 1e-4 #1E-4
MOMENTUM = 0.9

log_path = './graph/logs' # path to tensorboard graph
TFRECORD_NAME = 'smalldata.tfrecord'

# momentum to 0.9 and
# weight decay to 0.0005 (at this point we do not include)

'''
How to use Tensorboard
Run the command line:
> tensorboard --logdir=./graph/logs
'''

'''
TODO
1. data needs to be normlized 
2. make the test set smaller like 20 or 30
3. make the test set image size smaller

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
    tfrecord_filename = TFRECORD_NAME
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

    G_LEARNING_RATE = 1e-5 #1e-5
    LEARNING_RATE = 1e-4 #1e-4

    tf.set_random_seed(1111)
    tf.reset_default_graph()

    #calc step number
    step_num = int(6 / BATCH_SIZE)

    # place holders for shadow image input and shadow free image input
    with tf.variable_scope('Data_Input'):
        shadow = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3], name ='Shadow_image')
        shadow_free = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3], name = 'Shadow_free_image')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lr_g = tf.placeholder(tf.float32, name='learn_rate_g')
        lr = tf.placeholder(tf.float32, name ='learn_rate_as')
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
            opt_g = tf.train.MomentumOptimizer(lr_g, MOMENTUM)
            train_g_op = opt_g.minimize(model.loss, var_list = model.g_net_variables)
            opt_a = tf.train.MomentumOptimizer(lr, MOMENTUM)
            train_as_op = opt_a.minimize(model.loss, var_list = model.a_net_variables)
            opt_s = tf.train.MomentumOptimizer(lr, MOMENTUM)
            train_as_op = opt_s.minimize(model.loss, var_list = model.s_net_variables)
            train_op = tf.group(train_g_op,train_as_op)


        # loading dataset ...
        reader = tf.TFRecordReader()
        img_input, img_gt = read_tfrecord_pair_img(reader, TFRECORD_NAME, False)

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

		# parameter
        tf.summary.scalar('epoch',epoch)
        tf.summary.scalar('loss',model.loss)
        tf.summary.scalar('lr_as',lr)
        tf.summary.scalar('lr_g',lr_g)

		# images
        tf.summary.image('F_matte',model.shadow_matte)
        tf.summary.image('GT_matte',model.gt_shadow_matte)
        tf.summary.image('F_free',model.f_shadowfree)
        tf.summary.image('GT_free',model.gt_shadowfree)

        summary_op = tf.summary.merge_all()
        print('start training ...')
        writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())


        print('check if there is backup')
        if backupFlag and tf.train.get_checkpoint_state('./backup'):
            print('backup file loading ...')
            saver = tf.train.Saver()
            saver.restore(sess, './backup/latest')
        else:
            print('no backup ...')


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


                        _,loss,summary = sess.run([train_op,model.loss,summary_op],
                                            feed_dict={
                                            shadow: img_input_seq,
                                            shadow_free:  img_gt_seq,
                                            keep_prob: 0.5,
                                            lr: LEARNING_RATE,
                                            lr_g: G_LEARNING_RATE
                                            })
                        # print('loss: {}'.format(loss))
                        loss_val += loss
                        # write log
                        writer.add_summary(summary, (sess.run(epoch) -1)*step_num+i)


                        if i == int(step_num -1):
                            A, S,SM,GT_SM = sess.run([model.A_output,model.S_output,model.shadow_matte,model.gt_shadow_matte],
                                    feed_dict={
                                    shadow: img_input_seq,
                                    shadow_free:  img_gt_seq,
                                    keep_prob: 1.0,
                                    lr: LEARNING_RATE,
                                    lr_g:G_LEARNING_RATE
                                    })
                            A.reshape([BATCH_SIZE,224,224,3])
                            S.reshape([BATCH_SIZE,224,224,3])
                            SM.reshape([BATCH_SIZE,224,224,3])
                            GT_SM.reshape([BATCH_SIZE,224,224,3])
                            np.savetxt('./output/a_out/A{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), A[0,:,:,0], delimiter=',')   # X is an array
                            np.savetxt('./output/s_out/S{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), S[0,:,:,0], delimiter=',')   # X is an array
                            np.savetxt('./output/sm_out/SM{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), SM[0,:,:,0], delimiter=',')   # X is an array
                            np.savetxt('./output/sm_gt_out/GT_SM{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), GT_SM[0,:,:,0], delimiter=',')   # X is an array



                    print('total loss in one epoch: {}'.format(loss_val))

                    # check for validation
                    # find shadow matte
                    #f_shadow_free, gt_shadow_free = sess.run([model.f_shadowfree,model.gt_shadowfree],
                    #                                        feed_dict={
                    #                                        shadow: img_input_seq,
                    #                                        shadow_free: img_gt_seq,
                    #                                        keep_prob: 1.0,
                    #                                        lr: LEARNING_RATE,
                    #                                        lr_g:G_LEARNING_RATE
                    #                                        })
                    #np.savetxt('./output/f_shadow/f_shadow{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), f_shadow_free[0,:,:,:], delimiter=',')   # X is an array
                    #np.savetxt('./output/gt_shadow/gt_shadow{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), gt_shadow_free[0,:,:,:], delimiter=',')   # X is an array					
                    # find
                    #out = f_shadow_free[0].astype(np.uint8)
                    #gt = gt_shadow_free[0].astype(np.uint8)
                    #np.savetxt('./output/f_shadow/f_shadow{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), out[:,:,0], delimiter=',')
                    #np.savetxt('./output/gt_shadow/gt_shadow{}.out'.format("{0:06d}".format((sess.run(epoch) -1)*step_num* BATCH_SIZE+i)), gt[:,:,0], delimiter=',')

                    #image_show(out)
                    #image_show(gt)

                    #cv2.imwrite('./output/f_shadow/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                    #cv2.imwrite('./output/gt_shadow/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                    # saving data
                    saver = tf.train.Saver()
                    saver.save(sess, './backup/latest', write_meta_graph = False)
                    if sess.run(epoch) == EPOCH:
                        saver.save(sess,'./backup/fully_trained',write_meta_graph = False)

                    # decay learning rate
                    if (sess.run(epoch) % 200) == 0:
                        LEARNING_RATE = lr_decay(INIT_LEARNING_RATE, 1, sess.run(epoch))
                        G_LEARNING_RATE = lr_decay(INIT_G_LEARNING_RATE, 1, sess.run(epoch))
                        print('decreasing learning rate ...')


                else:
                    print('breaking out of while loop ...')
                    break

        except tf.errors.OutOfRangeError:
            print('ERROR: out of range\nDone Training -- epoch reached its limit')

        finally:
            print('FINAL: stop coordinate and joing thread')
            coord.request_stop()
            coord.join(threads)

def image_show(np_image):
    img = Image.fromarray(np_image,'RGB')
    img.show()

def lr_decay(lr_input, decay_rate,num_epoch):
    return lr_input / (1 + decay_rate*num_epoch)

if __name__ == '__main__':
    make_tfrecord()
    train(False)
