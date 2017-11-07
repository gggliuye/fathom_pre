# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import tensorflow.python.platform
import tensorflow as tf
import numpy as np

#from tensorflow.models.image.cifar10 import cifar10_input
import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'ecotaxa_data_ex/',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
  tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return cifar10_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=FLAGS.data_dir,
                              batch_size=FLAGS.batch_size)


def conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
      tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def fully_connected( x,in_dim, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [-1, in_dim])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.add(tf.matmul(x, w), b)

def inference(images):
    with tf.variable_scope('init'):
      x = conv('init_conv', images, 3, 1, 16, [1,1,1,1])

      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 4
    in_filter=16
    out_filter=16
    stride=[1,1,1,1]
    with tf.variable_scope('unit_1_0'):
      with tf.variable_scope('shared_activation'):
        #x = batch_norm('init_bn', x)
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

        x = tf.nn.relu(x)
        orig_x = x
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x

    with tf.variable_scope('unit_1_1'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x

    with tf.variable_scope('unit_1_2'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x

    with tf.variable_scope('unit_1_3'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x


    in_filter=16
    out_filter=32
    stride=[1,2,2,1]

    with tf.variable_scope('unit_1_4'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    in_filter=32
    out_filter=32
    stride=[1,1,1,1]
    with tf.variable_scope('unit_2_0'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
      with tf.variable_scope('sub_add'):
        x += orig_x

    with tf.variable_scope('unit_2_1'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x

    with tf.variable_scope('unit_2_2'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x

    in_filter=32
    out_filter=32
    stride=[1,2,2,1]

    with tf.variable_scope('unit_2_3'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    in_filter=32
    out_filter=32
    stride=[1,1,1,1]
    with tf.variable_scope('unit_2_4'):
      with tf.variable_scope('shared_activation'):
        orig_x = x
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
      with tf.variable_scope('sub1'):
        x = conv('conv1', x, 3, in_filter, out_filter, stride)

      with tf.variable_scope('sub2'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm3')
        x = tf.nn.relu(x)
        x = conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
  
      with tf.variable_scope('sub_add'):
        x += orig_x


    in_filter=32
    out_filter=64
    stride=[1,2,2,1]
    with tf.variable_scope('unit_3_0'):
      with tf.variable_scope('sub1'):
        x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        x = tf.nn.relu(x)
        x = conv('conv1', x, 3, in_filter, out_filter, stride)
    """
    with tf.variable_scope('conv2') as scope:
      kernel = tf.get_variable('weights', [5,5,64,64], initializer=tf.truncated_normal_initializer(stddev=1e-4))
      x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    """
    with tf.variable_scope('unit_last'):
      x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm_final')
      x = tf.nn.relu(x)
      x = tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,1,1,1],padding='VALID',name='avg_pool')
    
    with tf.variable_scope('logitss'):
      x = fully_connected(x,64 ,3)  
      predictions = tf.nn.softmax(x)
    return predictions


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  #indices = tf.reshape(range(FLAGS.batch_size), [FLAGS.batch_size, 1]) 
  concated = tf.concat([indices, sparse_labels],1)
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels = dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.contrib.deprecated.scalar_summary(l.op.name +' (raw)', l)
    tf.contrib.deprecated.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.contrib.deprecated.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.contrib.deprecated.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  """
  for grad, var in grads:
    if grad:
      tf.contrib.deprecated.histogram_summary(var.op.name + '/gradients', grad)
  """
  
  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.mkdir(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
