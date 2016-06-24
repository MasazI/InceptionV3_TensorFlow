# encoding: utf-8

import tensorflow as tf
import settings
FLAGS = settings.FLAGS


class DataSet:
    def __init__(self):
        pass

    def distort_color(self, image, thread_id=0):
        """Distort the color of the image.

        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.

        Args:
          image: Tensor containing single image.
          thread_id: preprocessing thread ID.
          scope: Optional scope for op_scope.
        Returns:
          color-distorted image
        """
        with tf.op_scope([image], 'distort_color'):
            color_ordering = thread_id % 2

            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)

            # The random_* ops do not necessarily clamp.
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def _generate_image_and_label_batch(self, image, label, min_queue_examples, batch_size, shuffle=True):
        '''
        imageとlabelのmini batchを生成
        '''
        num_preprocess_threads = FLAGS.num_threads
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * FLAGS.batch_size,
                min_after_dequeue=min_queue_examples
            )
            # Display the training images in the visualizer
            #tf.image_summary('images', images, max_images=batch_size)
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * FLAGS.batch_size,
                min_after_dequeue=min_queue_examples
            )

        return images, labels

    def cnt_samples(self, filepath):
        return sum(1 for line in open(filepath))

    def test_inputs(self, csv, batch_size):
        print("input csv file path: %s, batch size: %d" % (csv, batch_size))
        filename_queue = tf.train.string_input_producer([csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, label = tf.decode_csv(serialized_example, [["path"], [0]])

        label = tf.cast(label, tf.int32)
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
        print "original image shape:"
        print image.get_shape()

        # resize to distort
        dist = tf.image.resize_images(image, FLAGS.scale_h, FLAGS.scale_w)
        dist = tf.random_crop(image, [FLAGS.input_h, FLAGS.input_w, 3])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
        print (
        'filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

        return self._generate_image_and_label_batch(dist, label, min_queue_examples, batch_size, shuffle=False)


    def csv_inputs(self, csv, batch_size, distorted=False):
        print("input csv file path: %s, batch size: %d" % (csv, batch_size))
        filename_queue = tf.train.string_input_producer([csv], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, label = tf.decode_csv(serialized_example, [["path"], [0]])

        label = tf.cast(label, tf.int32)
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
        print "original image shape:"
        print image.get_shape()

        if distorted:
            # resize to distort
            dist = tf.image.resize_images(image, FLAGS.scale_h, FLAGS.scale_w)

            # random crop
            dist = tf.image.resize_image_with_crop_or_pad(dist, FLAGS.input_h, FLAGS.input_w)

            # random flip
            dist = tf.image.random_flip_left_right(dist)

            # color constancy
            dist = self.distort_color(dist)
        else:
            # resize to input
            dist = tf.image.resize_images(image, FLAGS.input_h, FLAGS.input_w)

        print "dist image shape:"
        print dist.get_shape()

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train * min_fraction_of_examples_in_queue)
        print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

        return self._generate_image_and_label_batch(dist, label, min_queue_examples, batch_size)


def debug(data):
    return data

if __name__ == "__main__":
    dataset = DataSet()
    images, labels = dataset.csv_inputs(FLAGS.tfcsv, FLAGS.batch_size)

    images_eval = debug(images)
    labels_eval = debug(labels)

    # initialization
    init = tf.initialize_all_variables()
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    images_val, labels_val = sess.run([images_eval, labels])

