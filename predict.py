# encoding: utf-8
import tensorflow as tf
import settings
FLAGS = settings.FLAGS

import os
import re
import copy
from datetime import datetime
import time
from datasets import DataSet
import datasets

from PIL import Image

import model
import train_operation
import slim.slim
import numpy as np

def train():
    with tf.Graph().as_default():
        # global step number
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        dataset = DataSet()

        # get training set
        print("The number of training images is: %d" % (dataset.cnt_samples(FLAGS.predictcsv)))
        csv_predict = FLAGS.predictcsv
        lines = dataset.load_csv(csv_predict)

        images_ph = tf.placeholder(tf.float32, [1, 229, 229, 3])

        num_classes = FLAGS.num_classes
        restore_logits = not FLAGS.fine_tune

        # inference
        logits = model.inference(images_ph, num_classes, for_training=True, restore_logits=restore_logits)


        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_all_summaries()

        # initialization
        init = tf.initialize_all_variables()

        # session
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        summary_writer = tf.train.SummaryWriter(
            FLAGS.train_dir,
            graph_def=sess.graph.as_graph_def(add_shapes=True))

        print("start to predict.")
        for step, line in enumerate(lines):
            pil_img = Image.open(line[0])
            pil_img = pil_img.resize((229, 229))
            img_array_r = np.asarray(pil_img)
            img_array = img_array_r[None, ...]
            softmax_eval = sess.run([logits[2]], feed_dict={images_ph: img_array})
            print("%s,%d" % (line, softmax_eval))
        print("finish to predict.")
        coord.request_stop()
        coord.join(threads)
        sess.close()


def test():
    # load settings file
    print(FLAGS.tfcsv)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir) and not FLAGS.fine_tune:
        print("Caution: train dir is already exists.")
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
