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
from time import localtime, strftime
import sys



# Load Data
pickle_file = 'data/genki_data.pkl'
with open(pickle_file, "rb") as input_file:
    data = pickle.load(input_file)

train_dataset_o = data['train_dataset'].reshape(-1, 256, 256, 1)
train_labels_o = data['train_labels']
train_masks_o = data['train_masks']
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
n_patches = int(sys.argv[1]) # goes from 2, 4, 8, 16, 32, 64

# dx = train_x.shape[1]
ds = train_labels.shape[1]
dx = image_size*image_size*image_depth

# set filename
filename = 'GENKI_'+str(n_patches)+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())

# Retore G
sess = tf.InteractiveSession()
# new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))

saver = tf.train.import_meta_graph('models/GENKI_pretrain.meta')
saver.restore(sess, tf.train.latest_checkpoint('models/'))
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("X:0")
S = graph.get_tensor_by_name("S:0")
G = graph.get_tensor_by_name("output/output:0")

file = open(filename+'_log.txt','w')
# finding privacy risk score per pixel for randomly-selected images
file.write('=== Finding privacy risk score per pixel ===\n')
number_patch_images = 8
# randidx = np.arange(0, number_patch_images)
randidx  =  [0, 2, 4, 8, 11, 16, 24, 32]
# randidx  =  [548, 1435, 1262, 1989, 235, 1469, 1965, 672, 539, 321]
n_pixels = int(image_size/n_patches)

file.write('Number of samples: {}, number of patches: {}, pixles  per dimension: {}\n'.format(number_patch_images, n_patches, n_pixels))
file.flush()

pixel_scores = np.zeros((number_patch_images, n_patches, n_patches, 1))
for l in range(number_patch_images):
    if l%1 == 0:
        file.write('Images: {}/{}\n'.format(l, number_patch_images))
        file.flush()
    idx = randidx[l]
    for j in range(n_patches):
        for k in range(n_patches):
            x_pixel = np.zeros((1, 256, 256, 1))
            # x_pixel[:, j*n_pixels:j*n_pixels+n_pixels, k*n_pixels:k*n_pixels+n_pixels, :] = train_dataset_o[idx, j*n_pixels:j*n_pixels+n_pixels, k*n_pixels:k*n_pixels+n_pixels, :]
            x_pixel[:, 0:j*n_pixels+n_pixels, 0:k*n_pixels+n_pixels, :] = train_dataset_o[idx, 0:j*n_pixels+n_pixels, 0:k*n_pixels+n_pixels, :]
            x_pixel = x_pixel.reshape((-1, image_size*image_size*num_channels)).astype(np.float32)

            # x_pixel_2 = np.concatenate((x_pixel, x_pixel), axis=0)
            s_pixel = train_labels[idx, :].reshape((1, 2))
            # s_pixel = np.identity(2)
            g_pixel = G.eval(feed_dict={X: x_pixel, S: s_pixel}, session=sess)

            pixel_scores[l, j, k, 0] = g_pixel[0, 1]


pickle_save_file = 'data/'+filename+'.pickle'
f = open(pickle_save_file, 'wb')
save = {
    'pixels': pixel_scores,
    'images': train_dataset_o[randidx],
    'randidx': randidx,
    'labels': train_labels_o[randidx],
    'masks': train_masks_o[randidx],
    }
pickle.dump(save, f, 2)
f.close()

# file.write('=== Start saving data ===\n')
# pickle_save_file = filename+'.pickle'
# f = open(pickle_save_file, 'wb')
# save = {
#     'G_train': G_train,
#     'G_test': G_test,
#     'psgx_train': psgx_train,
#     'psgx_test': psgx_test,
#     'train_labels': train_labels,
#     'test_labels': test_labels
#     }
# pickle.dump(save, f, 2)
# f.close()

file.write('=== Finished!!! ===\n')
file.flush()

file.close()
sess.close()
