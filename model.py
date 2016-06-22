# encoding: utf-8

import re
import tensorflow as tf
from slim import slim

import settings
FLAGS = settings.FLAGS

TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
  with tf.name_scope('summaries'):
    for act in endpoints.values():
      _activation_summary(act)

def inference(images, num_classes, for_training=False, restore_logits=True, scope=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': FLAGS.batchnorm_moving_average_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            # endpoints = {"logits", "predictions"}, predictions is softmax of logits
            logits, endpoints = slim.inception.inception_v3(
                images,
                dropout_keep_prob=0.8,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope)

        # Add summaries for viewing model statistics on TensorBoard.
        # _activation_summaries(endpoints)

        # Grab the logits associated with the side head. Employed during training.
        auxiliary_logits = endpoints['aux_logits']

    softmax = endpoints['predictions']

    print logits.get_shape()
    print softmax.get_shape()

    # logits: output of final layer, auxliary_logits: output of hidden layer, softmax: predictions
    return logits, auxiliary_logits, softmax



def loss(logits, labels, batch_size=None):
    """Adds all losses for the model.

    Note the final loss is not returned. Instead, the list of losses are collected
    by slim.losses. The losses are accumulated in tower_loss() and summed to
    calculate the total loss.

    Args:
    logits: List of logits from inference(). Each entry is a 2-D float Tensor.
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    batch_size: integer
    """
    if not batch_size:
        batch_size = FLAGS.batch_size

    # Reshape the labels into a dense Tensor of
    # shape [FLAGS.batch_size, num_classes].
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    sparse_labels = tf.cast(sparse_labels, tf.int32)
    concated = tf.concat(1, [indices, sparse_labels])
    num_classes = logits[0].get_shape()[-1].value
    dense_labels = tf.sparse_to_dense(concated,
                                    [batch_size, num_classes],
                                    1.0, 0.0)

    print "-"*10
    print type(logits)
    print len(logits)
    print logits[0].get_shape()
    print logits[1].get_shape()
    print "-"*10

    # Cross entropy loss for the main softmax prediction.
    slim.losses.cross_entropy_loss(logits[0],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=1.0)

    # Cross entropy loss for the auxiliary softmax head.
    slim.losses.cross_entropy_loss(logits[1],
                                 dense_labels,
                                 label_smoothing=0.1,
                                 weight=0.4,
                                 scope='aux_loss')
