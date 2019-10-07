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
        "tongue_label_raw": tf.FixedLenFeature([], tf.string),
        "vein_label_raw": tf.FixedLenFeature([], tf.string),
        # "index": tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    print(tf.shape(image))
    image = tf.reshape(image, shape=[320, 256, 3]) # i-f image is already has grayscale
    tongue_label = tf.decode_raw(parsed["tongue_label_raw"], tf.uint8)
    tongue_label = tf.cast(tongue_label, tf.float32)
    # tongue_label = tf.reshape(tongue_label, shape=[512, 640, 1])
    tongue_label = tf.reshape(tongue_label, shape=[320, 256, 1])

    vein_label = tf.decode_raw(parsed["vein_label_raw"], tf.uint8)
    vein_label = tf.cast(vein_label, tf.float32)
    # vein_label = tf.reshape(vein_label, shape=[512, 640, 1])
    vein_label = tf.reshape(vein_label, shape=[320, 256, 1])
    # label = tf.cast(parsed["label_raw"], tf.int32)
    # index = tf.cast(parsed["index"], tf.int32)
    return image, vein_label,tongue_label

def input_fn(filenames, train, n_repeat, batch_size, buffer_size=3):
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
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=20)  # parallelize reading

    # dataset to shuffle with specified samples from dataset


    print("dataset:",dataset)
    # print(dataset.graph)
    # print(tf.get_default_graph)
    dataset = dataset.map(parser, num_parallel_calls=20)
    print("dataset:", dataset)

    if train:
        # #---------------------------
        # print("start prerprocessing...")
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = n_repeat
        print(filenames)
        # rotate
        rotated_10_dataset = dataset.map(rotate_dataset_10_tongue_and_vein_RGB,num_parallel_calls=20)  # +25  /for val + 5 / for agu + 402
        rotated_20_dataset = dataset.map(rotate_dataset_20_tongue_and_vein_RGB, num_parallel_calls=20)  # +25  / +5 / +402
        rotated_30_dataset = dataset.map(rotate_dataset_30_tongue_and_vein_RGB, num_parallel_calls=20)  # +25  /+ 5 /+402
        #
        # # concate rotations
        dataset = dataset.concatenate(rotated_10_dataset) # num = 50    /10   /804
        dataset = dataset.concatenate(rotated_20_dataset) # num =75     /15   /1206
        dataset = dataset.concatenate(rotated_30_dataset) # num = 100    /20  / 1608

        # flip left an right
        fliped_left_right_dataset = dataset.map(flip_left_right_dataset_tongue_and_vein, num_parallel_calls=20)  # +100 /+ 20   /+1608
        dataset = dataset.concatenate(fliped_left_right_dataset)  # num=200      /40   /+ 3216

        # dataset range change to [0, 1]
        dataset = dataset.map(only_norm_tongue_and_vein, num_parallel_calls=20)  # (0, 1)
        # change to [-1, 1]
        dataset = dataset.map(from_zero_one_to_minus_positive_one_tongue_and_vein, num_parallel_calls=20)

        dataset = dataset.repeat(num_repeat)  #200 x num_repeat； if num_repeat =50, --> 200x50= 10000/ 40x50 =2000 / 3216x20 = 63200



        # generate batch dataset 1 epoch = 1272/2=636
        batch_dataset = dataset.batch(batch_size) # 200 x num_repeat/2 batch_size 10000(50)/2=5000 /---63200(20)/2 = 31600

        prefetch_dataset = batch_dataset.prefetch(buffer_size=10)

        # create iterator
        dataset_iterator = prefetch_dataset.make_one_shot_iterator()

        # # # get each batch data from iterator
        # batch_data = dataset_iterator.get_next()
        print(filenames, ":batch data iterator generation is done")
        return dataset_iterator
        # return batch_data

    else:
        num_repeat = 1

        # dataset = dataset.map(only_norm)

        # dataset range change to [0, 1]
        dataset = dataset.map(only_norm_tongue_and_vein, num_parallel_calls=20)  # (0, 1)
        # change to [-1, 1]
        dataset = dataset.map(from_zero_one_to_minus_positive_one_tongue_and_vein, num_parallel_calls=20)

        dataset = dataset.repeat(num_repeat)

        # generate batch dataset
        batch_dataset = dataset.batch(batch_size)

        # # dataset to shuffle with specified samples from dataset
        prefetch_dataset = batch_dataset.prefetch(buffer_size=10)

        # create iterator
        dataset_iterator = prefetch_dataset.make_one_shot_iterator()

        # # get each batch data from iterator
        # batch_data = dataset_iterator.get_next()
        print(filenames,":batch data iterator generation is done")
        print("iterator shapesssss.....................:", dataset_iterator.output_shapes)
        return dataset_iterator

