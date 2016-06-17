# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags train settings
flags.DEFINE_integer('batch_size', 30, 'the number of images in a batch.')
flags.DEFINE_string('tfcsv', 'data/tfcsv_train.txt', 'path to tf csv file for training.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train', "Directory where to write event logs and checkpoint")
tf.app.flags.DEFINE_integer('max_steps', 10000000, "Number of batches to run.")
tf.app.flags.DEFINE_string('subset', 'train', "Either 'train' or 'validation'.")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1, "How many GPUs to use.")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False, "is fine tune")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', "pretrained model's checkpoint path")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, "Initial learning rate.")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0, "Epochs after which learning rate decays.")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16, "Learning rate decay factor.")