import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy

np.set_printoptions(threshold=np.nan)


GPU = '0'   

#################################
# parameters
width  = 256
height = 256
channel = 3
epoch = 1000
latent_len = 100
train_data_dir = '../facades/train'
test_data_dir = '../facades/test'
val_data_dir = '../facades/val'
sample_dir = './samples'
checkpoint_dir = './checkpoint'
conv_kernel_size = 3

kernel_init = tf.truncated_normal_initializer(stddev=0.02)
#kernel_init=None
use_bias=False
padding='same'
bn_momentum=0.9
bn_eps = 0.00001
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

################################# 
# input
#G_in = tf.placeholder(tf.float32, (None, height, width, channel), name='g_input')
X = tf.placeholder(tf.float32, (None, height, width, channel), name='sketch_input')
PHASE = tf.placeholder(tf.bool, name='is_training')
Y = tf.placeholder(tf.float32, (None, height, width, channel), name='real_input')
global_step = tf.Variable(0, trainable=False, name='global_step')

#################################
# encode block (convolution)
def enc(in_layer, num_out_tensor, kernel_size=conv_kernel_size, strides=(2,2), padding='same' , use_bn=True, phase=True, relu=True):
    if use_bn:
        use_bias=False
    else: 
        use_bias=True
    _layer = tf.layers.conv2d(in_layer, num_out_tensor, kernel_size, strides=strides, padding=padding, activation=None, use_bias=use_bias, kernel_initializer=kernel_init)
    if use_bn:
        _layer = tf.layers.batch_normalization(_layer, training=phase, momentum=bn_momentum, epsilon=bn_eps) 
    if relu:
        _layer = tf.nn.relu(_layer)

    return _layer

#################################
# decode block (deconv)
def dec(in_layer, num_out_tensor, kernel_size=conv_kernel_size, strides=(2,2), padding='same' , use_bn=True, phase=True, relu=True):
    if use_bn:
        use_bias=False
    else:
        use_bias=True
    _layer = tf.layers.conv2d_transpose(in_layer, num_out_tensor, kernel_size, strides=strides, padding=padding, activation=None, use_bias=use_bias, kernel_initializer=kernel_init)
    if use_bn:
        _layer = tf.layers.batch_normalization(_layer, training=phase, momentum=bn_momentum, epsilon=bn_eps) 
    if relu:
        _layer = tf.nn.relu(_layer)

    return _layer

#################################
# generator
#with tf.name_scope('Generator'):
def G(input_image, is_training):
    #input image : 256x256x3
    with tf.variable_scope('G_variables'):
        l1 = enc(input_image, 64, phase=is_training, use_bn=False)#128x128x64
        l2 = enc(l1, 128, phase=is_training)         #64x64x128
        l3 = enc(l2, 256, phase=is_training)         #32x32x256
        l4 = enc(l3, 512, phase=is_training)         #16x16x512
        l5 = enc(l4, 512, phase=is_training)         #8x8x512
        l6 = enc(l5, 512, phase=is_training)         #4x4x512
        l7 = enc(l6, 512, phase=is_training)         #2x2x512
        l8 = enc(l7, 512, phase=is_training)         #1x1x512
        
        l9 = dec(l8, 512, phase=is_training, kernel_size=2)         #2x2x512
        skip_l7_l9 = tf.concat([l7, l9], axis=3)    
        l10 = dec(skip_l7_l9, 512, phase=is_training)                                      #4x4x512
        skip_l6_l10 = tf.concat([l6, l10], axis=3)
        l11 = dec(skip_l6_l10, 512, phase=is_training)                                     #8x8x512
        skip_l5_l11 = tf.concat([l5, l11], axis=3)
        l12 = dec(skip_l5_l11, 512, phase=is_training)                                     #16x16x512
        skip_l4_l12 = tf.concat([l4, l12], axis=3)
        l13 = dec(skip_l4_l12, 256, phase=is_training)                                     #32x32x256
        skip_l3_l13 = tf.concat([l3, l13], axis=3)
        l14 = dec(skip_l3_l13, 128, phase=is_training)                                     #64x64x128
        skip_l2_l14 = tf.concat([l2, l14], axis=3)
        l15 = dec(skip_l2_l14, 64, phase=is_training)                                     #128x128x64
        skip_l1_l15 = tf.concat([l1, l15], axis=3)
        l16 = dec(skip_l1_l15, 3, phase=is_training, use_bn=False, relu=False)            #256x256x3
        l17 = tf.nn.tanh(l16)
    return l17



#################################
# discriminator
#with tf.name_scope('Discriminator'):
def D(input_image, Y, is_training, reuse=False):
    with tf.variable_scope('D_variables') as scope:
        if reuse:
            scope.reuse_variables()
        input = tf.concat([input_image, Y], axis=3)     #256x256x6
        l1 = enc(input, 64, phase=is_training, use_bn=False)#128x128x64
        l2 = enc(l1, 128, phase=is_training)         #64x64x128
        l3 = enc(l2, 256, phase=is_training)         #32x32x256
        l4 = enc(l3, 512, phase=is_training, strides=(1,1), padding='valid')         #31x31x512
        l5 = enc(l4, 1,   phase=is_training, strides=(1,1), padding='valid', relu=False)         #30x30x1
       
    return l5


#with tf.name_scope('Optimizer'):

with tf.device("/device:GPU:{}".format(GPU)):
    #################################
    # gan network
    G = G(X, PHASE)
    D_real = D(X, Y, PHASE)
    D_fake = D(X, G, PHASE, reuse=True)

    #################################

    # train step
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    D_loss = D_loss_real + D_loss_fake

    G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
    G_loss_L1 = tf.reduce_mean(tf.abs(G-Y))
    G_gan_loss_ratio = 0.5
    G_loss = G_gan_loss_ratio*G_loss_gan + (1-G_gan_loss_ratio)*G_loss_L1


    D_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_variables')
    G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_variables')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        D_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=D_var_list, global_step = global_step)
        G_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=G_var_list, global_step = global_step)


#tensorboard
tf.summary.scalar('D_Loss', D_loss)
tf.summary.scalar('G_Loss', G_loss)

#################################
#misc

def get_batch_file_list(n):
    start = n*batch_size
    return train_data_file_list[start:(start+batch_size)]

def img_squeeze(img):   # 0~255 -->  -1 ~ 1
    return ((img*2.0)/256.0) -1.

def img_recover(img):
    img =((img+1.)*256.0)/2.0
    return img.astype(int)

def read_image(file, scale_w, scale_h):
    img = scipy.misc.imread(file, mode='RGB').astype(np.float)
    img = img_squeeze(img)
    # print(img)
    # seperate image (originals are concatted)
    return np.split(img, 2, axis=1)
    

def read_batch(batch_file_list):
    x = np.empty([0, height, width, channel], dtype='f')
    y = np.empty([0, height, width, channel], dtype='f')

    for file in batch_file_list:
        y_, x_ = read_image(file , width, height) 
        x_ = np.expand_dims(x_, axis=0) #[1, 256, 256, 3]
        y_ = np.expand_dims(y_, axis=0) #[1, 256, 256, 3]
        if not x.size:
            x = x_
            y = y_
        else:
            x = np.concatenate([x,x_], axis=0)
            y = np.concatenate([y,y_], axis=0)
        
    return x, y

#################################
# training

# data 
batch_size = 16
train_data_file_list = glob.glob(os.path.join(train_data_dir, '*.jpg'))
train_num_data = len(train_data_file_list)
train_num_batch = int(train_num_data/batch_size)

print ("train epoch     : %d"%epoch )
print ("train num data  : %d"%train_num_data )
print ("train num batch : %d"%train_num_batch )

test_size = 10  #num images to test
test_data_file_list = glob.glob(os.path.join(test_data_dir, '*.jpg'))
test_data_file_list = test_data_file_list[:test_size]
test_x, test_y = read_batch(test_data_file_list)
test_x_recon  = (test_x+1.0)/2.0
            



#session
gpu_options = tf.GPUOptions(allow_growth=True)  # Without this, the process occupies whole area of memory in the all GPUs.
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

#if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#    saver.restore(sess, ckpt.model_checkpoint_path)
#else:
#    sess.run(tf.global_variables_initializer())
sess.run(tf.global_variables_initializer())

#tensorboard
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

#epoch =1
#num_batch=100

for e in range(epoch):  
    random.shuffle(train_data_file_list)  #to avoid overfitting
    for b in range(train_num_batch):
        this_batch = get_batch_file_list(b)
        #print('File : ', this_batch)
        x_input, y_input = read_batch(this_batch)

        #train 
        is_training=True
        D_, D_batch_loss = sess.run([D_train, D_loss], feed_dict={X:x_input, Y:y_input, PHASE:is_training})
        G_, G_batch_loss = sess.run([G_train, G_loss], feed_dict={X:x_input, Y:y_input, PHASE:is_training})
        
        print("epoch: %04d"%e, "batch: %05d"%b, "D_loss: {:.04}".format(D_batch_loss),"G_loss: {:.04}".format(G_batch_loss) )

        
        #save input X
        save_input=False
        if save_input:
            fig, ax = plt.subplots(1, batch_size, figsize=(batch_size,1))
            for k in range(batch_size):
                ax[k].set_axis_off()
                #ax[k].imshow((x_input[k]/255.0))
                ax[k].imshow(((x_input[k]+1.0)/2.0))
            plt.savefig('samples/x.png', bbox_inches='tight')
            plt.close(fig)
        
        # testing
        #if not b%100: 
        if b==(train_num_batch-1):
            is_training=False
            samples = sess.run(G, feed_dict={X:test_x, Y:test_y, PHASE:is_training})
            samples = (samples+1.0)/2.0
            fig, ax = plt.subplots(2, test_size, figsize=(test_size, 2), dpi=400)
            for k in range(test_size):
                ax[0][k].set_axis_off()
                ax[0][k].imshow(test_x_recon[k])
                ax[1][k].set_axis_off()
                ax[1][k].imshow(samples[k])
            plt.savefig(sample_dir+'/pix2pix_{}'.format(str(e).zfill(3)) + '_{}.png'.format(str(b).zfill(5)), bbox_inches='tight')
            plt.close(fig)

    saver.save(sess, checkpoint_dir+'/pix2pix.ckpt', global_step=global_step)
    #tensorboard
    is_training=False
    summary = sess.run(merged, feed_dict={X:x_input, Y:y_input, PHASE:is_training})
    writer.add_summary(summary, global_step=sess.run(global_step))







