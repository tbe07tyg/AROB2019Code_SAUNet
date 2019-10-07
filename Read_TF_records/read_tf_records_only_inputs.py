import tensorflow as tf
from Preprocessing.pre_processing import *

def parser(record):
    """
    get images and labels out from tfrecords..
    :param record: tfrecords datafile
    :return: input image and input label
    """
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        # "label_raw": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    print(tf.shape(image))
    image = tf.reshape(image, shape=[320, 256, 3]) # i-f image is already has grayscale

    return image

def input_fn_input_only(filenames, batch_size):
    """

    :param runner: the current runner
    :param filenames:  the filenames of dataset : tf records name
    :param train:  whether this function is used to generate train dataset or test dataset
    :param n_repeat: how many repeats for your dataset
    :param buffer_size:  the buffer size of shuffling
    :return: iterator of specified tf records
    """
    print(filenames)
    # <editor-fold desc="Get batch data from tfrecords files">
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    print("dataset:",dataset)
    # print(dataset.graph)
    # print(tf.get_default_graph)
    dataset = dataset.map(parser)
    print("dataset:", dataset)



    num_repeat = 1

    # dataset = dataset.map(only_norm)

    # dataset range change to [0, 1]
    dataset = dataset.map(only_norm_only_input)  # (0, 1)
    # change to [-1, 1]
    dataset = dataset.map(from_zero_one_to_minus_positive_one_only_input)

    dataset = dataset.repeat(num_repeat)

    # generate batch dataset
    batch_dataset = dataset.batch(batch_size)

    # # dataset to shuffle with specified samples from dataset
    # prefetch_dataset = batch_dataset.shuffle(buffer_size=buffer_size)

    # create iterator
    dataset_iterator =  batch_dataset.make_one_shot_iterator()

    # # get each batch data from iterator
    # batch_data = dataset_iterator.get_next()
    print(filenames,":batch data iterator generation is done")
    print("iterator shapesssss.....................:", dataset_iterator.output_shapes)
    return dataset_iterator

