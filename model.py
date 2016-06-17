# encoding: utf-8

import re
import tensorflow as tf
from slim import slim

import settings
FLAGS = settings.FLAGS

def inference(images, num_classes, for_training=False, restore_logits=True, scope=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': FLAGS.batchnorm_moving_average_decay,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
