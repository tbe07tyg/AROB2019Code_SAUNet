import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import pickle
import json
import time
def discrim_conv(batch_input, out_channels, stride):
    # [ batch , 256 , 256 , 6 ] ===>[ batch , 258 , 258 , 6 ]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    '''
    [0,0]: 第一维batch大小不扩充
    [1,1]：第二维图像宽度左右各扩充一列，用0填充
    [1,1]：第三维图像高度上下各扩充一列，用0填充
    [0,0]：第四维图像通道不做扩充
    '''
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                            kernel_initializer=tf.random_normal_initializer(0, 0.02))


# - generator -conv
def gen_conv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                kernel_initializer=initializer)


# - generator -deconv
def gen_deconv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same",
                                          depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same",
                                          kernel_initializer=initializer)


# define "LReLu"-leakage relu activation function
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


# batch normalization
def batchnorm(inputs, training=True):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=training,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def batchnorm_with_name(inputs, training=True, name="bn"):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=training,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02), name=name)

# Show all variables in current model
def show_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def show_op_names(model):
    print("\n[ print op list ]\n")
    for op in model.sess.graph.get_operations():
        print(op.name)

    print("\n[ op list is done ]\n")

def show_op_names_with_grpah(graph):
    print("\n[ print op list ]\n")
    for op in graph.get_operations():
        print(op.name)

    print("\n[ op list is done ]\n")

def show_tensor_names_with_grpah(graph):
    print("\n[ print tensor list ]\n")
    for op in graph.as_graph_def().node:
        print(op.name)

    print("\n[ tensor list is done ]\n")

def show_tensor_names(model):
    print("\n[ print op list ]\n")
    for tensor in model.sess.graph.as_graph_def().node:
        print(tensor)

    print("\n[ op list is done ]\n")



def checkFolders(dir_list):
    for dir in list(dir_list):
        if not os.path.exists(dir):
            os.makedirs(dir)

def childs(t, d=0):
    print ('-' * d, t.name)
    for child in t.op.inputs:
        childs(child, d + 1)

def storeTree(inputTree, dir, filename):
    """
    :param inputTree: the data object need to be written
    :param filename:  the saved filename
    :return:
    """
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)
    filename=dir+filename
    print(filename)
    fw = open(filename, 'wb') #以二进制读写方式打开文件
    pickle.dump(inputTree, fw)  #pickle.dump(对象, 文件，[使用协议])。序列化对象
    # 将要持久化的数据“对象”，保存到“文件”中，使用有3种，索引0为ASCII，1是旧式2进制，2是新式2进制协议，不同之处在于后者更高效一些。
    #默认的话dump方法使用0做协议
    fw.close() #关闭文件

def JSON_storeTree(inputTree, dir, filename):
    """
    :param inputTree: the data object need to be written
    :param filename:  the saved filename
    :return:
    """
    if os.path.exists(dir):
        pass
    else:
        os.makedirs(dir)
    filename=dir+filename
    print(filename)
    with open(filename , 'wb') as fw:
        pickle.dump(inputTree, fw)

def JSON_grabTree(dir, filename):
    """
    :param filename: the file data to be read
    :return:
    """
    filename = dir + filename
    with open(filename, 'r') as filehandle:
         content =json.load(filehandle.read())
    return content #读取文件，反序列化

def grabTree(dir, filename):
    """
    :param filename: the file data to be read
    :return:
    """
    filename = dir + filename
    fr = open(filename, 'rb')
    return pickle.load(fr) #读取文件，反序列化

def grabTree_full_address(full_address):
    """
    :param filename: the file data to be read
    :return:
    """

    fr = open(full_address, 'rb')
    return pickle.load(fr) #读取文件，反序列化
