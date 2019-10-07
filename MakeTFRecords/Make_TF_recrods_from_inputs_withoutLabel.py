# imports
import tensorflow as tf
import glob
import sys
from PIL import Image
import numpy as np
import os
from random import shuffle

# define feature types converters
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x) for x in value]))

# load image from file names list
def load_image(addr):
    # read an image and resize to (512, 512)
    # cv2 load images as BGR, convert it to RGB
    img = Image.open(addr)
    img.convert("RGB")
    print("---------------------------------------------------------------------------",img.size)
    if img is None:
        return None
    # # img = cv2.resize(img, (605, 600), interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img resize half to avoid om problem of up
    # img = img.resize((640, 512))
    img = img.resize((256, 320))  # FOR NEW DATA
    print(img.size)
    img = np.array(img) # this operation will change rgb with channel 3 , gray with channel 1

    print("kdsajfljka;", img.shape)
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    print(img.shape)

    return img

def write_tf_records_tongue_vein_pack(out_filename, addrs, tongue_labels, vein_labels):
    """
        this function is to use to create data records
        :param out_filename: output file name
        :param addrs: input images address
        :param labels: labels of input images
        :return:
        """
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1 images
        if not i % 1:
            print("input data: {}/{}".format(i + 1, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

        if img is None:
            continue

        if isinstance(tongue_labels[i], int):
            tongue_label = tongue_labels[i]
            vein_label = vein_labels[i]
            print("int")
            # create a feature
            feature = {
                'image_raw': _bytes_feature(img.tostring()),
                'tongue_label_raw': _int64_feature(tongue_label),
                'vein_label_raw': _int64_feature(vein_label)
            }
        else:
            print("image")
            img_tongue_label = load_image(tongue_labels[i])
            img_vein_label = load_image(vein_labels[i])
            print("image shape:", img.shape)
            print("tongue_label shape:", img_tongue_label.shape)
            print("vein_label shape:", img_vein_label.shape)
            feature = {
                'image_raw': _bytes_feature(img.tostring()),
                'tongue_label_raw': _bytes_feature(img_tongue_label.tostring()),
                'vein_label_raw': _bytes_feature(img_vein_label.tostring())
                # 'index': _int64_feature(i)
            }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    print("current write finished -------------------->")
    sys.stdout.flush()


def write_tf_records_vein_inputs_only(out_filename, addrs):
    """
        this function is to use to create data records
        :param out_filename: output file name
        :param addrs: input images address
        :param labels: labels of input images
        :return:
        """
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1 images
        if not i % 1:
            print("input data: {}/{}".format(i + 1, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

        if img is None:
            continue

        if isinstance(addrs[i], int):
            print("int")
            # create a feature
            feature = {
                'image_raw': _bytes_feature(img.tostring()),

            }
        else:
            print("image")
            print("image shape:", img.shape)
            feature = {
                'image_raw': _bytes_feature(img.tostring()),
            }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    print("current write finished -------------------->")
    sys.stdout.flush()


# 去除文件的后缀，获取文件名
def get_name(path):
    # os.path.basename(),返回path最后的文件名。若path以/或\结尾，那么就会返回空值。
    # os.path.splitext(),分离文件名与扩展名；默认返回(fname,fextension)元组
    name, _ = os.path.splitext(os.path.basename(path))
    print(name)
    return name


def sort_paths(paths): # get the filename without suffix
    if all(get_name(path).isdigit() for path in paths):
        sorted_paths = sorted(paths, key=lambda path: int(get_name(path)))
    else:
        sorted_paths = sorted(paths)
    return sorted_paths


def make_tf_records():
    train_input_path = "../Data/ALL_data/AugmentedInputs/*"




    # for inputs
    train_input_addrs = glob.glob(train_input_path)  # return a list of strings which are the absolute path of each images in the given folder

    # sort file paths
    train_inputs_addrs = sort_paths(train_input_addrs)



    print("sorted addrs:")
    print("inputs:")
    print(train_inputs_addrs, "\n")



    # # package inputs and labels and shuffle
    # train_c = list(zip(train_inputs_addrs))  # zip. 打包 inputs and labels filename path
    # print("--------------------------:",train_c)
    shuffle(train_inputs_addrs)  # shuffle 打乱顺序

    #
    # unpack shuffled file names
    # (train_inputs_addrs)= zip(* train_c )  # un-zip. 打乱后解包
    print("--------------------------:", train_inputs_addrs)
    write_tf_records_vein_inputs_only("train_vein_inputs_only.tfrecords", train_inputs_addrs)


if __name__ == '__main__':
    make_tf_records()