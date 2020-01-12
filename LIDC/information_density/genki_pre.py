# -*- coding: utf-8 -*-
""""
Codes for NeurIPS'19 on Privacy Watchdog
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import tensorflow as tf
import pickle
import gzip
import numpy as np
# import scipy as sp
# import pandas as pd
from time import localtime, strftime

# from util import *

# Load Data
pickle_file = 'data/genki_data.pkl'
with open(pickle_file, "rb") as input_file:
    data = pickle.load(input_file)

train_dataset_o = data['train_dataset']
train_labels_o = data['train_labels']
valid_dataset_o = data['valid_dataset']
valid_labels_o = data['valid_labels']
test_dataset_o = data['test_dataset']
test_labels_o = data['test_labels']

# transform data into tensorflow-friendly format
num_labels = 2
image_size = 256
pixel_depth = 255.0
image_depth = 1
num_channels = image_depth # = 3 (RGB)
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size*num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset_o, train_labels_o)
valid_dataset, valid_labels = reformat(valid_dataset_o, valid_labels_o)
test_dataset, test_labels = reformat(test_dataset_o, test_labels_o)




# Parameters
N = train_dataset.shape[0] # number of samples
EPOCH_psgx = 150
EPOCH_DV = 100
lr = 5e-3
mb = 256
# n_patches = int(sys.argv[1]) # goes from 2, 4, 8, 16, 32, 64

# dx = train_x.shape[1]
ds = train_labels.shape[1]
dx = image_size*image_size*image_depth


# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)

# def lrelu(x, alpha=0.2):
#     return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# G_W1 = tf.Variable(xavier_init([ds+dx, num_neuron]), name='G_W1')
# G_b1 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b1')
# G_W2 = tf.Variable(xavier_init([num_neuron, num_neuron]), name='G_W2')
# G_b2 = tf.Variable(tf.zeros(shape=[num_neuron]), name='G_b2')
# G_W3 = tf.Variable(xavier_init([num_neuron, ds]), name='G_W3')
# G_b3 = tf.Variable(tf.zeros(shape=[ds]), name='G_b3')
# theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]
#
# W1 = tf.Variable(xavier_init([dx, num_neuron]), name='W1')
# b1 = tf.Variable(tf.zeros(shape=[num_neuron]), name='b1')
# W2 = tf.Variable(xavier_init([num_neuron, num_neuron]), name='W2')
# b2 = tf.Variable(tf.zeros(shape=[num_neuron]), name='b2')
# W3 = tf.Variable(xavier_init([num_neuron, ds]), name='W3')
# b3 = tf.Variable(tf.zeros(shape=[ds]), name='b3')
# theta_sgx = [W1, b1, W2, b2, W3, b3]

def g_approx(x, s, ds, _IMAGE_SIZE, _IMAGE_CHANNELS, _NUM_CLASSES):
    # _IMAGE_SIZE = 64
    # _IMAGE_CHANNELS = 3
    # _NUM_CLASSES = 2
    _RESHAPE_SIZE = 4*4*512*16

    with tf.name_scope('data'):
        # x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        # y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 1, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv1', conv1)
    # tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv2', conv2)
    # tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv3', conv3)
    # tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    # with tf.variable_scope('conv4') as scope:
    #     kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    #     conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv4 = tf.nn.relu(pre_activation, name=scope.name)
    # # tf.summary.histogram('Convolution_layers/conv4', conv4)
    # # tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))
    #
    # with tf.variable_scope('conv5') as scope:
    #     kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    #     conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv5 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv5', conv5)
    # tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        # print(s.shape)
        # print(pool3.shape)
        # print(reshape.shape)
        reshape = tf.concat([s, reshape], axis=1)
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.tanh(tf.matmul(reshape, weights) + biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/fc1', local3)
    # tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.tanh(tf.matmul(local3, weights) + biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/fc2', local4)
    # tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, ds], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [ds], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    # tf.summary.histogram('Fully connected layers/output', softmax_linear)

    return softmax_linear
# def g_approx(x, s):
#     G_list = []
#     s = tf.cast(s, tf.float32)
#     x = tf.cast(x, tf.float32)
#     sx = tf.concat([s, x], 1)
#     fc1 = tf.nn.tanh(tf.matmul(sx, G_W1) + G_b1)
#     fc2 = tf.nn.tanh(tf.matmul(fc1, G_W2) + G_b2)
#     g = tf.matmul(fc2, G_W3) + G_b3
#     return g

def psgx_ps_approx(x, ds, _IMAGE_SIZE, _IMAGE_CHANNELS, _NUM_CLASSES):
    # _IMAGE_SIZE = 64
    # _IMAGE_CHANNELS = 3
    # _NUM_CLASSES = 2
    _RESHAPE_SIZE = 4*4*512*16

    with tf.name_scope('data'):
        # x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        # y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope('conv1_2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 1, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2_2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3_2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))
    #
    # with tf.variable_scope('conv4_2') as scope:
    #     kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    #     conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv4 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv4', conv4)
    # tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))
    #
    # with tf.variable_scope('conv5_2') as scope:
    #     kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
    #     conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv5 = tf.nn.relu(pre_activation, name=scope.name)
    # tf.summary.histogram('Convolution_layers/conv5', conv5)
    # tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))
    #
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    
    with tf.variable_scope('fully_connected1_2') as scope:
        reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2_2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output_2') as scope:
        weights = variable_with_weight_decay('weights', [192, ds], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [ds], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        # print(softmax_linear.shape)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    psgx = tf.nn.softmax(softmax_linear)
    ps_vec = tf.reshape(tf.reduce_mean(psgx, axis=0), (ds, 1))
    # print(f"softmax_linear: {softmax_linear.shape}")
    # print(f"psgx: {psgx.shape}")
    # print(f"ps_vec: {ps_vec.shape}")
    
    return psgx, softmax_linear, ps_vec
# def psgx_ps_approx(x, dx, ds):
#     fc1 = lrelu(tf.matmul(x, W1) + b1)
#     fc2 = lrelu(tf.matmul(fc1, W2) + b2)
#     psgx_logits = tf.matmul(fc2, W3) + b3
#     psgx = tf.nn.softmax(psgx_logits)
#     ps_vec = tf.reshape(tf.reduce_mean(psgx, axis=0), (ds, 1))
#     return psgx, psgx_logits, ps_vec

def joint_mean_sx(G, psgx):
    return tf.reduce_mean(tf.reduce_sum(tf.multiply(G, psgx), 1), 0)

def marginal_mean_sx(G, ps_vec):
    return tf.reduce_mean(tf.matmul(tf.exp(G), ps_vec))

def mi_sx(x, s, psgx, ps_vec):
    G = g_approx(x, s, ds, image_size, image_depth, num_labels)
    clip_min = np.float32(-3)
    clip_max = np.float32(3)
    G_clip = tf.clip_by_value(G, clip_min, clip_max)

    sup_loss = joint_mean_sx(G, psgx) - tf.log(marginal_mean_sx(G_clip, ps_vec))
    return sup_loss, G

def train(x, s, mb_size, EPOCH_psgx, EPOCH_DV, x_test, s_test, filename):
    file = open(filename+'_log.txt','w')

    file.write('=== Dataset Summary ===\n')
    file.write('Training set: {}, {}\n'.format(train_dataset.shape, train_labels.shape))
    file.write('Validation set: {}, {}\n'.format(valid_dataset.shape, valid_labels.shape))
    file.write('Test set: {}, {}\n'.format(test_dataset.shape, test_labels.shape))
    file.flush()

    n_train = train_dataset.shape[0]
    n_test = test_dataset.shape[0]

    file.write('=== Parameter Summary ===\n')
    file.write('dx = {}, ds = {}, n_train = {}, n_test = {}\n'.format(dx, ds, n_train, n_test))
    file.write('EPOCH_psgx: {}, EPOCH_DV: {}, learning_rate: {}, batch_size: {}\n'.format(EPOCH_psgx, EPOCH_DV, lr, mb_size))
    file.flush()



    sess = tf.InteractiveSession()

    # Placeholders
    X = tf.placeholder(tf.float32, [None, dx], name='X')
    S = tf.placeholder(tf.float32, [None, ds], name='S')
    Psgx = tf.placeholder(tf.float32, [None, ds])
    Ps_vec = tf.placeholder(tf.float32, [ds, 1])

    # Compute pygx
    psgx, psgx_logits, ps_vec = psgx_ps_approx(X, ds, image_size, image_depth, num_labels)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=S, logits=psgx_logits))
    solver_psgx = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)

    # Compute mutual information
    MI_SX, G = mi_sx(X, S, Psgx, Ps_vec)
    DV_loss = -1*MI_SX
    solver_DV = tf.train.AdagradOptimizer(lr).minimize(DV_loss)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # Compute pygx
    file.write('=== Training Psgx... ===\n')
    file.write('Epoch\t batch\t Train Loss\t Val Loss\n')
    file.flush()
    for epoch in range(EPOCH_psgx):
        for m in range(int(n_train/mb_size)):
            x_mb = x[m*mb_size:m*mb_size+mb_size]
            s_mb = s[m*mb_size:m*mb_size+mb_size]
            _, current_loss = sess.run([solver_psgx, cross_entropy], feed_dict={X: x_mb, S: s_mb})
            if epoch % 10 == 0:
                val_loss = cross_entropy.eval(feed_dict={X: valid_dataset, S: valid_labels}, session=sess)
                file.write('{}\t {}\t {:.4f}\t {:.4f}\n'.format(epoch, m, current_loss, val_loss))
                file.flush()

    psgx_train = psgx.eval(feed_dict={X: x}, session=sess)
    ps_vec_train = ps_vec.eval(feed_dict={X: x}, session=sess)

    psgx_val = psgx.eval(feed_dict={X: valid_dataset}, session=sess)
    ps_vec_val = ps_vec.eval(feed_dict={X: valid_dataset}, session=sess)

    file.write('psgx_train: {}\n'.format(psgx_train.shape))

    file.write('Finish Training Psgx\n')
    file.write('=== Training MI... ===\n')
    file.write('Epoch\t batch\t Loss\n')
    file.flush()
    for epoch in range(EPOCH_DV):
        for m in range(int(n_train/mb_size)):
            x_mb = x[m*mb_size:m*mb_size+mb_size]
            s_mb = s[m*mb_size:m*mb_size+mb_size]
            psgx_train_mb = psgx_train[m*mb_size:m*mb_size+mb_size]
            # ps_vec_train_mb = ps_vec_train[m*mb_size:m*mb_size+mb_size]
            _, current_DV_loss = sess.run([solver_DV, DV_loss], feed_dict={X: x_mb, S: s_mb, Psgx: psgx_train_mb, Ps_vec: ps_vec_train})
            if epoch % 10 == 0:
                # val_loss = DV_loss.eval(feed_dict={X: valid_dataset, S: valid_labels, Psgx: psgx_val, Ps_vec: ps_vec_val}, session=sess)
                # file.write('{}\t {}\t {:.4f}\t {:.4f}\n'.format(epoch, m, -1*current_DV_loss, -1*val_loss))
                file.write('{}\t {}\t {:.4f}\n'.format(epoch, m, -1*current_DV_loss))
                file.flush()


    MI_SX_train = MI_SX.eval(feed_dict={X: x, S: s, Psgx: psgx_train, Ps_vec: ps_vec_train}, session=sess)
    G_train = G.eval(feed_dict={X: x, S: s, Psgx: psgx_train, Ps_vec: ps_vec_train}, session=sess)

    psgx_test = psgx.eval(feed_dict={X: x_test}, session=sess)
    ps_vec_test = ps_vec.eval(feed_dict={X: x_test}, session=sess)
    MI_SX_test = MI_SX.eval(feed_dict={X: x_test, S: s_test, Psgx: psgx_test, Ps_vec: ps_vec_test}, session=sess)
    G_test = G.eval(feed_dict={X: x_test, S: s_test, Psgx: psgx_test, Ps_vec: ps_vec_test}, session=sess)

    correct_prediction = tf.equal(tf.argmax(S, axis=1), tf.argmax(Psgx, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc_train = sess.run(accuracy, feed_dict={S: s, Psgx: psgx_train})
    acc_test = sess.run(accuracy, feed_dict={S: s_test, Psgx: psgx_test})

    file.write('=== Results Summary ===\n')
    file.write('Training Accuracy: {}\n'.format(acc_train))
    file.write('Test Accuracy: {}\n'.format(acc_test))
    file.write('MI_SX_train = {}\n'.format(MI_SX_train))
    file.write('MI_SX_test = {}\n'.format(MI_SX_test))
    file.flush()

    # Checkpoint
    saver.save(sess, 'models/'+filename)

    file.write('=== Finished!!! ===\n')
    file.flush()

    file.close()
    sess.close()
    return


## creates and saves tensorflow models for each of the digits
if __name__ == '__main__':
    filename = 'GENKI_pretrain'
    train(train_dataset, train_labels, mb, EPOCH_psgx, EPOCH_DV, test_dataset, test_labels, filename)
