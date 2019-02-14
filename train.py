import numpy as np
import tensorflow as tf
from utils.utils import *
import utils.mnist_data
from vae import *
from tensorflow.examples.tutorials.mnist import input_data


######################################
######### Necessary Flags ############
######################################
tf.app.flags.DEFINE_string(
    'train_root', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_root',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
                            'Maximum number of checkpoints that TensorFlow will keep.')

##########################################
############## Model Flags ###############
##########################################
tf.app.flags.DEFINE_boolean(
    'add_noise', False, 'Boolean for adding salt & pepper noise to input image')

tf.app.flags.DEFINE_float(
    'learn_rate', 0.001, 'learn rate for Adam optimizer.')

tf.app.flags.DEFINE_integer('z_dim', 20,
                            'Dimension of latent vector')

tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch Size')

tf.app.flags.DEFINE_integer('num_epochs', 20,
                            'Number of epochs for training.')

tf.app.flags.DEFINE_integer('n_hidden', 500,
                            'Number of hidden units in MLP')


########################################
# Store all elemnts in FLAG structure! #
########################################

FLAGS = tf.app.flags.FLAGS

################################################
################# handling errors!##############
################################################

if not os.path.isabs(FLAGS.train_root):
    raise ValueError('You must assign absolute path for --train_root')

if not os.path.isabs(FLAGS.checkpoint_root):
    raise ValueError('You must assign absolute path for --checkpoint_root')


#######################################
############## Training ###############
#######################################

with tf.Session() as sess:
    mnist_vae = VAE(sess=sess,
                    epoch=FLAGS.num_epochs,
                    batch_size=FLAGS.batch_size,
                    z_dim=FLAGS.z_dim,
                    dataset_name="mnist",
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    result_dir="./result_dir",
                    log_dir=FLAGS.log_dir
                    )
    mnist_vae.build_model()
    mnist_vae.train()