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

import model
import train_operation
import slim.slim
import numpy as np

def train():
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        dataset = DataSet()

        # get trainsets
        print("The number of train images: %d", (dataset.cnt_samples(FLAGS.tfcsv)))
        images, labels = dataset.csv_inputs(FLAGS.tfcsv, FLAGS.batch_size, distorted=True)

        images_debug = datasets.debug(images)

        # get testsets
        #test_cnt = dataset.cnt_samples(FLAGS.testcsv)
        test_cnt = 100
	#test_cnt = 5
        print("The number of train images: %d", ())
        images_test, labels_test = dataset.test_inputs(FLAGS.testcsv, test_cnt)

        images_test_debug = datasets.debug(images_test)

        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        num_classes = FLAGS.num_classes
        restore_logits = not FLAGS.fine_tune

        # inference
        # logits is tuple (logits, aux_liary_logits, predictions)
        # logits: output of final layer, auxliary_logits: output of hidden layer, softmax: predictions
        logits = model.inference(images, num_classes, for_training=True, restore_logits=restore_logits)
        logits_test = model.inference(images_test, num_classes, for_training=False, restore_logits=restore_logits, reuse=True, dropout_keep_prob=1.0)

        # loss
        model.loss(logits, labels, batch_size=FLAGS.batch_size)
        model.loss_test(logits_test, labels_test, batch_size=test_cnt)
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
        losses_test = tf.get_collection(slim.losses.LOSSES_COLLECTION_TEST)

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
        #total_loss = tf.add_n(losses, name='total_loss')
        total_loss_test = tf.add_n(losses_test, name='total_loss_test')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])
        loss_averages_test = tf.train.ExponentialMovingAverage(0.9, name='avg_test')
        loss_averages_op_test = loss_averages_test.apply(losses_test + [total_loss_test])

        print "="*10
        print "loss length:"
        print len(losses)
        print len(losses_test)
        print "="*10

        # for l in losses + [total_loss]:
        #     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        #     # session. This helps the clarity of presentation on TensorBoard.
        #     loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
        #     # Name each loss as '(raw)' and name the moving average version of the loss
        #     # as the original loss name.
        #     tf.scalar_summary(loss_name + ' (raw)', l)
        #     tf.scalar_summary(loss_name, loss_averages.average(l))

        # loss to calcurate gradients
        #
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        tf.scalar_summary("loss", total_loss)

        with tf.control_dependencies([loss_averages_op_test]):
            total_loss_test = tf.identity(total_loss_test)
        tf.scalar_summary("loss_eval", total_loss_test)

        # Reuse variables for the next tower.
        #tf.get_variable_scope().reuse_variables()

        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

        # Retain the Batch Normalization updates operations only from the
        # final tower. Ideally, we should grab the updates from all towers
        # but these stats accumulate extremely fast so we can ignore the
        # other stats from the other towers without significant detriment.
        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        # add input summaries
        # summaries.extend(input_summaries)

        # train_operation and operation summaries
        train_op = train_operation.train(total_loss, global_step, summaries, batchnorm_updates)

        # trainable variables's summary
        #for var in tf.trainable_variables():
        #    summaries.append(tf.histogram_summary(var.op.name, var))

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation from the last tower summaries.
        #summary_op = tf.merge_summary(summaries)
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

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, logits_eval, loss_value, labels_eval, images_debug_eval = sess.run([train_op, logits[0], total_loss, labels, images_debug])
            duration = time.time() - start_time

            dataset.output_images(images_debug_eval, "debug", "train")

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('train %s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

            if step % 100 == 0:
                print("predict:")
                print type(logits_eval)
                print logits_eval.shape
                print logits_eval.argmax(1)
                print("target:")
                print labels_eval
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                test_start_time = time.time()
                logits_test_eval, total_loss_test_val, labels_test_eval, images_test_debug_eval = sess.run([logits_test[0], total_loss_test, labels_test, images_test_debug])
                test_duration = time.time() - test_start_time

                dataset.output_images(images_test_debug_eval, "debug_test", "test")

                print("test predict:")
                print type(logits_test_eval)
                print logits_test_eval.shape
                print logits_test_eval.argmax(1)
                print("test target:")
                print labels_test_eval
                test_examples_per_sec = test_cnt / float(test_duration)
                format_str_test = ('test %s: step %d, loss = %.2f, (%.1f examples/sec; %.3f sec/batch)')
                print(format_str_test % (datetime.now(), step, total_loss_test_val, test_examples_per_sec, test_duration))

                # Save the model checkpoint periodically.
                if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

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
