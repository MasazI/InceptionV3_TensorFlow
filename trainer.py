# encoding: utf-8
import tensorflow as tf
import settings
FLAGS = settings.FLAGS

def train():
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # get datsets


        # inference

        # loss

        # train operation
        

        # summary


        # train

def test():
    # load settings file
    print(FLAGS.tfcsv)


def main(argv=None):
    train()



if __name__ == '__main__':
    tf.app.run()