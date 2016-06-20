# encoding: utf-8
import tensorflow as tf
import settings
FLAGS = settings.FLAGS

import re
import copy
from datasets import DataSet

import model
import train_operation
import slim.slim

def train():
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # get datsets
        dataset = DataSet()
        images, labels = dataset.csv_inputs(FLAGS.tfcsv, FLAGS.batch_size)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        num_classes = FLAGS.num_classes
        restore_logits = not FLAGS.fine_tune

        # inference
        logits = model.inference(images, num_classes, for_training=True,
                                     restore_logits=restore_logits)
        # loss
        model.loss(logits, labels, batch_size=FLAGS.batch_size)
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on TensorBoard.
            loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.scalar_summary(loss_name + ' (raw)', l)
            tf.scalar_summary(loss_name, loss_averages.average(l))

        # loss to calcurate gradients
        with tf.control_dependencies([loss_averages_op]):
            loss = tf.identity(total_loss)


        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Retain the Batch Normalization updates operations only from the
        # final tower. Ideally, we should grab the updates from all towers
        # but these stats accumulate extremely fast so we can ignore the
        # other stats from the other towers without significant detriment.
        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        # train_operation
        train_op = train_operation.train(loss, global_step, summaries, batchnorm_updates)

        # summary
        summaries.extend(input_summaries)

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