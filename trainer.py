# encoding: utf-8
import tensorflow as tf
import settings
FLAGS = settings.FLAGS

from datasets import DataSet

import model
import slim.slim

def train():
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # get datsets
        dataset = DataSet()
        images, labels = dataset.csv_inputs(FLAGS.tfcsv, FLAGS.batch_size)

        # inference



        # loss

        # train operation


        # summary


        # train

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