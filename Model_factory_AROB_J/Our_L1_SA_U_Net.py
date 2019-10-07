import os
from MakeTFRecords.Make_TF_recrods_from_inputs_withoutLabel_forAutoencoder import *

from Utils.Utils_functions import *
from Read_TF_records.read_tf_records import *

import matplotlib.pyplot as plt
import tensorlayer as tl

import numpy as np
np.set_printoptions(threshold=np.inf)
import json


# plt.ion(x, )
class Our_L1_SA_U_Net(object):
    """
    This is the SA-UNet main code
    """

    # initialize model
    def __init__(self):
        # Read keys/values from flags and assign to self
        # for key, val in flags.__dict__.items():
        #     if key not in self.__dict__.keys():
        #         self.__dict__[key] = val

        # Build graph for network model
        self.initialize_parameters()

        # Create tfrecords if file does not exist
        if not os.path.exists(os.path.join(self.data_dir, 'train_tongue_vein.tfrecords')):
            print(os.path.exists(os.path.join(self.data_dir, 'train_tongue_vein.tfrecords')))
            print(os.path.join(self.data_dir, 'train_tongue_vein.tfrecords'))
            print("\n [ Creating tfrecords files ]\n")
            make_tf_records()

        # Initialize training and validation datasets
        self.initialize_dataset()

        # Define tensor for updating global step
        self.global_step = tf.train.get_or_create_global_step()
        self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)


        #
        print("building network...")
        self.build_model()
        print("network was built")
    def initialize_parameters(self):
        # Define training switch initial = True
        self.training = True

        # 和模型训练相关的参数
        # self.train_beta_gamma = True
        self.weight_decay = 0.0001

        #
        self.model_name = "SA_UNET_L1"

        self.num_epoch = 100 # Total number of epoches
        self.batch_size = 2  # Training batch size

        self.num_batches_per_epoch = 100  # Total number of batches for one epoch

        self.num_batches = self.num_batches_per_epoch*self.num_epoch # Total number of batches

        self.adam_beta1 = 0.9  # Adam optimizer beta1 parameter
        self.data_dir = "../Model_factory_AROB_J/"  # Directory containing tfrecords
        self.display_step = 1  # Step count for displaying progress
        self.summary_step = 100  # Step count for saving summaries

        self.raw_shape = [320, 256]  # Resolution to use when saving generated images

        self.log_dir = "./" + self.model_name + "/logs/"  # Directory for saving log files
        self.checkpoint_dir = "./" + self.model_name + "/Checkpoints/"  # Directory for saving checkpoints"
        # self.checkpoint_test_dir = "../Model_factory/" + self.model_name + "/Checkpoints/"  # "Directory for test checkpoints pb generation"
        self.test_evaluation_dir = "./Saved_Test_Evaluation_Data/"  # "Directory for test data storeage"
        self.IoU_tracking_saved_dir= "./IOU_tracking_Data/"  # "Directory for test data storeage"
        self.loss_saved_dir = "./Losses_tracking_Data/"  # "Directory for loss data storeage"

        self.plot_step = 250  # Step count for saving plots of generated images
        self.plot_dir = "./" + self.model_name + "/predictions/"  # Directory for saving plots of generated images

        self.name = self.model_name  # Model Name
        self.learning_rt = 0.00015  # Initial learning rate  raw 1e-3 to large
        self.lr_decay_step = 2000  # Learning rate decay step
        self.lr_decay_rate = 0.8 # Learning rate decay rate

        self.step = 0
        self.momentum = 0.5  # 0.9 will be dead
        self.power = 0.9


        # for the loss weight

        self.output_entropy_weight = 1
        self.output_l1_weight = 1
        self.dice_weight = 2




    def initialize_dataset(self):
        print(self.data_dir)
        train_filenames = os.path.join(self.data_dir, 'train_tongue_vein.tfrecords')
        val_filenames = os.path.join(self.data_dir, 'val_tongue_vein.tfrecords')
        test_filenames = os.path.join(self.data_dir, 'test_tongue_vein.tfrecords')
        print(train_filenames)
        # print(x, val_filenames)
        # print(x, test_filenames)
        # total dataset size: x, training: 25, validating 5, testing 22： in total ,training batches = 5000, validation
        self.train_batch_data_iterator = input_fn(filenames=train_filenames, train=True, n_repeat=self.num_epoch,
                                                  buffer_size=60, batch_size=self.batch_size)
        self.val_batch_data_iterator = input_fn(filenames=val_filenames, train=True, n_repeat=self.num_epoch * 5,
                                                buffer_size=20, batch_size=self.batch_size)
        self.test_batch_data_iterator = input_fn(filenames=test_filenames, train=False, n_repeat=self.num_epoch,
                                                 buffer_size=20, batch_size=self.batch_size)

    #   should give iteraotor!!!!!!!!

    def set_session(self, sess):
        self.sess = sess

    # Reinitialize handles for datasets iterators when restoring from checkpoint
    def reinitialize_handles(self):
        self.train_handle = self.sess.run(self.train_batch_data_iterator.string_handle())
        self.validation_handle = self.sess.run(self.val_batch_data_iterator.string_handle())
        self.test_handle = self.sess.run(self.test_batch_data_iterator.string_handle())

    # define model architecture ---------------------------------------------------!>
    # model used function  -------------------
    def _sep_conv2d_with_bn_relu(self, input_tensor, num_filters, kernel_size, training,
                                 strides=(1, 1), dilation_rate=(1, 1),
                                 use_bias=True, name="_sep_conv2d_with_bn_relu"):
        with tf.name_scope(name):
            conv = tf.layers.separable_conv2d(inputs=input_tensor,
                                              filters=num_filters,
                                              kernel_size=kernel_size,
                                              padding="same",
                                              activation=None,
                                              strides=strides,
                                              dilation_rate=dilation_rate,
                                              depthwise_initializer=tf.variance_scaling_initializer(),
                                              pointwise_initializer=tf.variance_scaling_initializer(),
                                              use_bias=use_bias,
                                              name=name)  # kernel_initializer: An initializer for the convolution kernel.
            # conv = tf.layers.batch_normalization(conv, momentum=0.9997, epsilon=0.0001, scale=True, training=training,
            #                                      trainable=train_bn,
            #                                      name=name + "_bn")  # batch normalization momentum = decay

            conv1_batch_norm = tf.layers.batch_normalization(conv, axis=3, epsilon=1e-5, momentum=0.1, training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02), name=name + "_bn")  # batch normalization
            conv = tf.nn.relu(conv1_batch_norm, name=name + "_bn" + "_relu")
            return conv

    def _sep_conv2d_with_bn_tanh(self, input_tensor, num_filters, kernel_size, training,
                                 strides=(1, 1), dilation_rate=(1, 1),
                                 use_bias=True, name="_sep_conv2d_with_bn_tanh"):
        conv = tf.layers.separable_conv2d(inputs=input_tensor,
                                          filters=num_filters,
                                          kernel_size=kernel_size,
                                          padding="same",
                                          activation=None,
                                          strides=strides,
                                          dilation_rate=dilation_rate,
                                          depthwise_initializer=tf.variance_scaling_initializer(),
                                          pointwise_initializer=tf.variance_scaling_initializer(),
                                          use_bias=use_bias,
                                          name=name)  # kernel_initializer: An initializer for the convolution kernel.
        # conv = tf.layers.batch_normalization(conv, momentum=0.9997, epsilon=0.0001, scale=True, training=training,
        #                                      trainable=train_bn,
        #                                      name=name + "_bn")  # batch normalization momentum = decay

        conv1_batch_norm = tf.layers.batch_normalization(conv, axis=3, epsilon=1e-5, momentum=0.1, training=training,
                                                         gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                        0.02), name=name + "_bn")  # batch normalization
        conv = tf.nn.tanh(conv1_batch_norm, name=name + "_bn" + "_tanh")
        return conv

    def conv_1x1_block(self, input_tensor, num_filters_list, kernel_size_list, training,name, stride=(1,1), onexone_activation=None, use_bias=True):
        with tf.name_scope(name+"_Conv_1x1_block"):
            conv = self._sep_conv2d_with_bn_relu(input_tensor=input_tensor,
                                                 num_filters=num_filters_list[0],
                                                 kernel_size=kernel_size_list[0],
                                                 training = training,
                                                 strides=stride,
                                                 use_bias=use_bias,
                                                 name=name + "_conv")
            #
            conv_1x1 = tf.layers.conv2d(inputs=conv,
                                        filters=num_filters_list[1],
                                        kernel_size=kernel_size_list[1],
                                        padding="same", strides=stride,
                                        activation=onexone_activation,
                                        kernel_initializer=tf.initializers.truncated_normal(),
                                        name=name + "_conv_1x1")

            return conv_1x1

    def conv_double_block(self, input_tensor, num_filters_list, kernel_size_list, training, name, stride=(1, 1),
                       onexone_activation=None, use_bias=True):
        with tf.name_scope(name + "_Conv_1x1_block"):
            conv = self._sep_conv2d_with_bn_relu(input_tensor=input_tensor,
                                                 num_filters=num_filters_list[0],
                                                 kernel_size=kernel_size_list[0],
                                                 training=training,
                                                 strides=stride,
                                                 use_bias=use_bias,
                                                 name=name + "_conv")
            conv2 = self._sep_conv2d_with_bn_relu(input_tensor=conv,
                                                 num_filters=num_filters_list[1],
                                                 kernel_size=kernel_size_list[1],
                                                 training=training,
                                                 strides=stride,
                                                 use_bias=use_bias,
                                                 name=name + "_conv2")


            return conv2

    def PIM_with_GP(self, input_tensor, training):
        conv_2x2_1x1 = self.conv_1x1_block(input_tensor=input_tensor,
                                                num_filters_list=[256, 256],
                                                kernel_size_list=[[2, 2], [1, 1]],
                                                training=training,
                                                onexone_activation=None,
                                                name="conv_2x2_1x1")
        conv_3x3_1x1 = self.conv_1x1_block(input_tensor=input_tensor,
                                           num_filters_list=[256, 256],
                                           kernel_size_list=[[3, 3], [1, 1]],
                                           training=training,
                                           onexone_activation=None,
                                           name="conv_3x3_1x1")

        conv_4x4_1x1 = self.conv_1x1_block(input_tensor=input_tensor,
                                           num_filters_list=[256, 256],
                                           kernel_size_list=[[4, 4], [1, 1]],
                                           training=training,
                                           onexone_activation=None,
                                           name="conv_4x4_1x1")
        conv_5x5_1x1 = self.conv_1x1_block(input_tensor=input_tensor,
                                           num_filters_list=[256, 256],
                                           kernel_size_list=[[5, 5], [1, 1]],
                                           training=training,
                                           onexone_activation=None,
                                           name="conv_5x5_1x1")

        # # calculate global_pooling
        # conv_1x1 = tf.layers.conv2d(inputs=input_tensor,
        #                             filters=256,
        #                             kernel_size=[1, 1],
        #                             padding="same", strides=stride,
        #                             activation=tf.nn.relu,
        #                             kernel_initializer=tf.variance_scaling_initializer(),
        #                             name=name + "_input_conv_1x1")
        # GP_conv_2x2_1x1 =  tf.reduce_mean(conv_2x2_1x1, axis=[1, 2],
        #                                    keepdims=True, name="GP_conv_2x2_1x1")
        # GP_conv_3x3_1x1 = tf.reduce_mean(conv_3x3_1x1, axis=[1, 2],
        #                                  keepdims=True, name="GP_conv_3x3_1x1")
        # GP_conv_4x4_1x1 = tf.reduce_mean(conv_4x4_1x1, axis=[1, 2],
        #                                  keepdims=True, name="GP_conv_4x4_1x1")
        # GP_conv_5x5_1x1 = tf.reduce_mean(conv_5x5_1x1, axis=[1, 2],
        #                                  keepdims=True, name="GP_conv_5x5_1x1")
        #
        # # multiply
        # conv_2x2_1x1_multiply_gp =  tf.multiply(GP_conv_2x2_1x1,conv_2x2_1x1, name="conv_2x2_1x1_multiply_gp")
        # conv_3x3_1x1_multiply_gp = tf.multiply(GP_conv_3x3_1x1, conv_3x3_1x1, name="conv_3x3_1x1_multiply_gp")
        # conv_4x4_1x1_multiply_gp = tf.multiply(GP_conv_4x4_1x1 , conv_4x4_1x1, name="conv_4x4_1x1_multiply_gp")
        # conv_5x5_1x1_multiply_gp = tf.multiply(GP_conv_5x5_1x1, conv_5x5_1x1, name="conv_5x5_1x1_multiply_gp")



        concat_PIM_feature =  tf.concat([conv_2x2_1x1, conv_3x3_1x1,
                                         conv_4x4_1x1 ,conv_5x5_1x1], axis=-1, name="concat_PIM_feature")

        return concat_PIM_feature

    def conv_1x1_trans_concat_block(self, input1, input2, num_filters_list, kernel_size_list, name, stride=(1,1), onexone_activation=None):
        """
        :param input1: target dimension input
        :param input2:  smaller dimension input
        :param num_filters_list:  [target 1x1, samller 1x1 , smaller1x1 withTranspose]
        :param kernel_size_list:
        :param training:
        :param name:
        :param stride:
        :param onexone_activation:
        :param use_bias:
        :return:
        """

        with tf.name_scope(name+"_conv_1x1_trans_concat_block"):
            target_conv_1x1 = tf.layers.conv2d(inputs=input1,
                                        filters=num_filters_list[0],
                                        kernel_size= kernel_size_list[0],
                                        padding="same", strides=stride,
                                        activation=onexone_activation,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name=name + "_target_conv_1x1")
            smaller_conv_1x1 = tf.layers.conv2d(inputs=input2,
                                        filters=num_filters_list[1],
                                        kernel_size=kernel_size_list[1],
                                        padding="same", strides=stride,
                                        activation=onexone_activation,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name=name + "_smaller_conv_1x1")

            smaller_conv_1x1_transposed = tf.layers.conv2d_transpose(smaller_conv_1x1, num_filters_list[2], kernel_size=4, strides=(2, 2),
                                                               padding="same",
                                                               kernel_initializer=tf.variance_scaling_initializer(),
                                                               name=name + "_smaller_conv_1x1_transposed")

            # concat target 1x1 and smaller transposed
            concat_feature = tf.concat([target_conv_1x1, smaller_conv_1x1_transposed], axis=-1, name=name+"concat_feature")

            return concat_feature

    def conv_predict_trans_concat_block(self, smaller_feature, larger_feature, training, num_filters_list, kernel_size_list, name,feature_need=False):
        """
        :param input1: target dimension input
        :param input2:  smaller dimension input
        :param num_filters_list:  [smaller feature conv3x3 , smaller predict transpose withTranspose]
        :param kernel_size_list:  the same for kernel list
        :param training:
        :param name:
        :param stride:
        :param onexone_activation:
        :param use_bias:
        :return:
        """

        with tf.name_scope(name+"_conv_predict_trans_concat_block"):
            smaller_feature_conv3x3 = self._sep_conv2d_with_bn_relu(input_tensor=smaller_feature,
                                                                    num_filters=num_filters_list[0],
                                              kernel_size=kernel_size_list[0], training=training,
                                              name=name+"_smaller_feature_conv3x3")
            smaller_predict = tf.layers.conv2d(inputs=smaller_feature_conv3x3, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"_smaller_predict")

            smaller_predict_transposed = tf.layers.conv2d_transpose(smaller_predict, num_filters_list[1],
                                                                    kernel_size=kernel_size_list[1], strides=(2, 2),
                                                               padding="same",
                                                               kernel_initializer=tf.variance_scaling_initializer(),
                                                               name=name+"_smaller_predict_transposed")

            # concat target 1x1 and smaller transposed
            concat_feature = tf.concat([larger_feature, smaller_predict_transposed], axis=-1, name=name+"concat_feature")
            if feature_need ==True:
                return smaller_feature_conv3x3, smaller_predict, concat_feature
            else:
                return smaller_predict, concat_feature

    def conv_predict_trans_add_block(self, smaller_feature, larger_feature, training, num_filters_list,
                                        kernel_size_list, name, feature_need=False):
        """
        :param input1: target dimension input
        :param input2:  smaller dimension input
        :param num_filters_list:  [smaller feature conv3x3 , smaller predict transpose withTranspose]
        :param kernel_size_list:  the same for kernel list
        :param training:
        :param name:
        :param stride:
        :param onexone_activation:
        :param use_bias:
        :return:
        """

        with tf.name_scope(name + "_conv_predict_trans_concat_block"):
            smaller_feature_conv3x3 = self._sep_conv2d_with_bn_relu(input_tensor=smaller_feature,
                                                                    num_filters=num_filters_list[0],
                                                                    kernel_size=kernel_size_list[0], training=training,
                                                                    name=name + "_smaller_feature_conv3x3")
            smaller_predict = tf.layers.conv2d(inputs=smaller_feature_conv3x3, filters=1, kernel_size=[1, 1],
                                               padding="SAME",
                                               strides=(1, 1),
                                               activation=tf.nn.tanh,
                                               kernel_initializer=tf.variance_scaling_initializer(),
                                               use_bias=True,
                                               name=name + "_smaller_predict")

            smaller_predict_transposed = tf.layers.conv2d_transpose(smaller_predict, num_filters_list[1],
                                                                    kernel_size=kernel_size_list[1], strides=(2, 2),
                                                                    padding="same",
                                                                    kernel_initializer=tf.variance_scaling_initializer(),
                                                                    name=name + "_smaller_predict_transposed")

            # concat target 1x1 and smaller transposed
            add_feature = tf.add(larger_feature, smaller_predict_transposed, name=name+"add_smamller_pred_and_larger_feature")
            if feature_need == True:
                return smaller_feature_conv3x3, smaller_predict, add_feature
            else:
                return smaller_predict,  add_feature

    def conv_feature_trans_concat_block(self, smaller_feature, larger_feature, training, num_filters_list, kernel_size_list, name,feature_need=False):
        """
        :param input1: target dimension input
        :param input2:  smaller dimension input
        :param num_filters_list:  [smaller feature conv3x3 , smaller predict transpose withTranspose]
        :param kernel_size_list:  the same for kernel list
        :param training:
        :param name:
        :param stride:
        :param onexone_activation:
        :param use_bias:
        :return:
        """

        with tf.name_scope(name+"_conv_predict_trans_concat_block"):
            smaller_feature_conv3x3 = self._sep_conv2d_with_bn_relu(input_tensor=smaller_feature,
                                                                    num_filters=num_filters_list[0],
                                              kernel_size=kernel_size_list[0], training=training,
                                              name=name+"_smaller_feature_conv3x3")
            smaller_predict = tf.layers.conv2d(inputs=smaller_feature_conv3x3, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"_smaller_predict")

            smaller_feature_transposed = tf.layers.conv2d_transpose(smaller_feature_conv3x3, num_filters_list[1],
                                                                    kernel_size=kernel_size_list[1], strides=(2, 2),
                                                               padding="same",
                                                               kernel_initializer=tf.variance_scaling_initializer(),
                                                               name=name+"_smaller_feature_transposed")

            # concat target 1x1 and smaller transposed
            concat_feature = tf.concat([larger_feature, smaller_feature_transposed], axis=-1, name=name+"concat_feature")
            if feature_need ==True:
                return smaller_feature_conv3x3, smaller_predict, concat_feature
            else:
                return smaller_predict, concat_feature


    def trans_concat_block(self, input1, input2, num_filters,name):
        """
        :param input1: target dimension input
        :param input2:  smaller dimension input
        :param num_filters_list:  [target 1x1, samller 1x1 , smaller1x1 withTranspose]
        :param kernel_size_list:
        :param training:
        :param name:
        :param stride:
        :param onexone_activation:
        :param use_bias:
        :return:
        """

        with tf.name_scope(name+"_conv_1x1_trans_concat_block"):
            smaller_conv_1x1_transposed = tf.layers.conv2d_transpose(input2, num_filters, kernel_size=4, strides=(2, 2),
                                                               padding="same",
                                                               kernel_initializer=tf.variance_scaling_initializer(),
                                                               name=name + "_smaller_conv_1x1_transposed")

            # concat target 1x1 and smaller transposed
            concat_feature = tf.concat([input1, smaller_conv_1x1_transposed], axis=-1, name=name+"concat_feature")

            return concat_feature

    def _separable_conv_block(self, input_tensor, filters_list, kernel_size_list, name_list, training,
                              first_stride=(2, 2), last_stride=(2, 2), drop_rate=0.5, drop_bool=False):
        """

        :param input_tensor:
        :param filters_list:
        :param kernel_size_list: list in list [[1,1], [[2,2], [3,3],[4,4], [5,5]], [[2,2], [3,3],[4,4], [5,5]]]
        :param name_list:
        :param training:
        :param first_stride:
        :param last_stride:
        :param rate:  how much rate will be dropout default=0.5 means dropout all means throuway 0.5
        :return:
        """
        with tf.name_scope("Conv_block"):
            # short cut
            conv1x1_stride2_shortcut = tf.layers.conv2d(inputs=input_tensor, filters=filters_list[0],
                                                        kernel_size=kernel_size_list[0],
                                                        padding="same", strides=first_stride,
                                                        kernel_initializer=tf.variance_scaling_initializer()
                                                        , name=name_list[0])
            # seperatable convs
            print(conv1x1_stride2_shortcut.shape)
            x = self._sep_conv2d_with_bn_relu(input_tensor=input_tensor,
                                              num_filters=filters_list[1],
                                              kernel_size=kernel_size_list[1], training=training,
                                              name=name_list[1])
            x = self._sep_conv2d_with_bn_relu(input_tensor=x, num_filters=filters_list[2],
                                              kernel_size=kernel_size_list[2], training=training,
                                              name=name_list[2])
            x = self._sep_conv2d_with_bn_relu(input_tensor=x, num_filters=filters_list[3], strides=last_stride,
                                              kernel_size=kernel_size_list[3], training=training,
                                              name=name_list[3])
            print(x.shape)
            # add
            add_return = tf.add(conv1x1_stride2_shortcut, x, name=name_list[4])

            # drop out
            if drop_bool == False:
                pass
            else:
                add_return = tf.layers.dropout(inputs=add_return, rate=drop_rate, training=training)
            return add_return

    def _separable_conv_block_firstStride2(self, input_tensor, filters_list, kernel_size_list, name_list, training,
                              first_stride=(2, 2), last_stride=(2, 2), drop_rate=0.5, drop_bool=False):
        """

        :param input_tensor:
        :param filters_list:
        :param kernel_size_list: list in list [[1,1], [[2,2], [3,3],[4,4], [5,5]], [[2,2], [3,3],[4,4], [5,5]]]
        :param name_list:
        :param training:
        :param first_stride:
        :param last_stride:
        :param rate:  how much rate will be dropout default=0.5 means dropout all means throuway 0.5
        :return:
        """
        with tf.name_scope("Conv_block"):
            # short cut
            conv1x1_stride2_shortcut = tf.layers.conv2d(inputs=input_tensor, filters=filters_list[0],
                                                        kernel_size=kernel_size_list[0],
                                                        padding="same", strides=first_stride,
                                                        kernel_initializer=tf.variance_scaling_initializer()
                                                        , name=name_list[0])
            # seperatable convs
            print(conv1x1_stride2_shortcut.shape)
            x = self._sep_conv2d_with_bn_relu(input_tensor=input_tensor,
                                              num_filters=filters_list[1],strides=last_stride,
                                              kernel_size=kernel_size_list[1], training=training,
                                              name=name_list[1])
            x = self._sep_conv2d_with_bn_relu(input_tensor=x, num_filters=filters_list[2],
                                              kernel_size=kernel_size_list[2], training=training,
                                              name=name_list[2])

            print(x.shape)
            # add
            add_return = tf.add(conv1x1_stride2_shortcut, x, name=name_list[4])

            # drop out
            if drop_bool == False:
                pass
            else:
                add_return = tf.layers.dropout(inputs=add_return, rate=drop_rate, training=training)
            return add_return


    def _separable_identity_block(self, input_tensor, filters_list, kernel_size_list, name_list, training):
        """

        :param input_tensor:
        :param filters_list:
        :param kernel_size_list: list in list [[1,1], [[2,2], [3,3],[4,4], [5,5]], [[2,2], [3,3],[4,4], [5,5]]]
        :param name_list:
        :param training:
        :param first_stride:
        :param last_stride:
        :param rate:  how much rate will be dropout default=0.5 means dropout all means throuway 0.5
        :return:
        """
        with tf.name_scope("identity_block"):


            # seperatable convs

            x = self._sep_conv2d_with_bn_relu(input_tensor=input_tensor,
                                              num_filters=filters_list[0],
                                              kernel_size=kernel_size_list[0], training=training,
                                              name=name_list[0])
            x = self._sep_conv2d_with_bn_relu(input_tensor=x,
                                              num_filters=filters_list[1],
                                              kernel_size=kernel_size_list[1], training=training,
                                              name=name_list[1])
            x = self._sep_conv2d_with_bn_relu(input_tensor=x,
                                              num_filters=filters_list[2],
                                              kernel_size=kernel_size_list[2], training=training,
                                              name=name_list[2])

            return x


    # def f1(self,tensor, zero_tensor, name="switch"):
    #     return tf.multiply(tensor, zero_tensor, name=name+"_Switch_Off")

    def f1(self, tensor, zero_tensor, name="switch"):
        return tf.multiply(tensor, zero_tensor, name=name + "_Switch_Off")

    # def f2(self,tensor, name="switch"):
    #     return tf.identity(tensor, name=name + "_NoSwitch")
    #
    def f2(self,tensor, name="switch"):
        return tf.identity(tensor, name=name + "_SwitchOn")

    def switchTensor(self, tensor, zero_tensor, switchon_bool, name):
        """
        :param tensor:
        :return:
        """
        condition = switchon_bool
        print(condition)
        print(tensor.shape)
        print(zero_tensor.shape)
        out_tensor = tf.cond(condition, lambda: self.f2(tensor=tensor, name=name),
                             lambda: self.f1(tensor=tensor, zero_tensor=zero_tensor, name=name))
        print(out_tensor.shape)
        return out_tensor



    def f1_True(self):
        return tf.constant(True, dtype=tf.bool)

    def f2_False(self):
        return tf.constant(False, dtype=tf.bool)

    def trainingSwitch(self, training):
        condition = tf.constant(training, tf.bool)
        return tf.cond(condition, lambda: self.f1_True(),
                             lambda: self.f2_False())


    def predict_block(self, stage_input1, stage_input2, stage_input3, training, num_filters, name, feature_need=False):

        # zeros_tensor = tf.zeros([self.batch_size,stage_input1.shape[1], stage_input1.shape[2], stage_input1.shape[3]])
        # print(stage_input1.shape)

        x_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input1, num_filters=num_filters,
                                                kernel_size=[3, 3], training=training,
                                                name=name+"_x_3x3")
        xu_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input2, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name+"_xu_3x3")
        rx_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input3, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name+"_rx_3x3")
        # print(x_2_3x3.shape)
        x_2_3x3_predict = tf.layers.conv2d(inputs=x_2_3x3, filters=1, kernel_size=[1, 1],
                                           padding="SAME",
                                           strides=(1, 1),
                                           activation=tf.nn.tanh,
                                           kernel_initializer=tf.variance_scaling_initializer(),
                                           use_bias=True,
                                           name=name+"x_3x3_predict")
        # print(x_2_3x3_predict.shape)
        xu_2_3x3_predict = tf.layers.conv2d(inputs=xu_2_3x3, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"xu_3x3_predict")
        rx_2_3x3_predict = tf.layers.conv2d(inputs=rx_2_3x3, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"rx_3x3_predict")

        # stage2_predict_concat = tf.concat([x_2_predict_drop, xu_2_predict_drop, rx_2_predict_drop], axis=-1,
        #                                   name=name + "_predict_concat")
        #
        # stage2_predict = tf.reduce_mean(stage2_predict_concat, axis=-1, name=name+"_predict_reduce_mean", keepdims=True)
        # print(x_2_predict_drop.shape)

        x_2_3x3_predict_1x1 = tf.layers.conv2d(inputs=x_2_3x3_predict, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"x_2_3x3_predict_1x1")
        xu_2_3x3_predict_1x1 = tf.layers.conv2d(inputs=xu_2_3x3_predict, filters=1, kernel_size=[1, 1],
                                               padding="SAME",
                                               strides=(1, 1),
                                               kernel_initializer=tf.variance_scaling_initializer(),
                                               use_bias=True,
                                               name=name + "xu_2_3x3_predict_1x1")
        rx_2_3x3_predict_1x1 = tf.layers.conv2d(inputs=rx_2_3x3_predict, filters=1, kernel_size=[1, 1],
                                                padding="SAME",
                                                strides=(1, 1),
                                                kernel_initializer=tf.variance_scaling_initializer(),
                                                use_bias=True,
                                                name=name + "rx_2_3x3_predict_1x1")
        print(x_2_3x3_predict_1x1)
        print(xu_2_3x3_predict_1x1)
        print(rx_2_3x3_predict_1x1)
        stage2_predict=tf.add(tf.add(x_2_3x3_predict_1x1, xu_2_3x3_predict_1x1), rx_2_3x3_predict_1x1, name=name+"_add_predict")

        if feature_need ==False:
            return stage2_predict, x_2_3x3_predict, xu_2_3x3_predict, rx_2_3x3_predict
        else:
            return stage2_predict, x_2_3x3_predict, xu_2_3x3_predict, rx_2_3x3_predict, x_2_3x3 ,  xu_2_3x3,  rx_2_3x3

    def stage_predict_block(self, stage_input1, stage_input2, stage_input3,training, mask, num_filters, name):



        x_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input1, num_filters=num_filters,
                                                kernel_size=[3, 3], training=training,
                                                name=name+"_x_3x3")
        xu_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input2, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name+"_xu_3x3")
        rx_2_3x3 = self._sep_conv2d_with_bn_relu(input_tensor=stage_input3, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name+"_rx_3x3")


        x_2_3x3_add_mask =  tf.add(x_2_3x3, mask, name="x_2_3x3_add_mask")
        xu_2_3x3_add_mask = tf.add(xu_2_3x3, mask, name="xu_2_3x3_add_mask")
        rx_2_3x3_add_mask = tf.add(rx_2_3x3, mask, name="rx_2_3x3_add_mask")

        x_2_3x3_conv2= self._sep_conv2d_with_bn_relu(input_tensor=x_2_3x3_add_mask , num_filters=num_filters,
                                                kernel_size=[3, 3], training=training,
                                                name=name + "_x_3x3_conv2")
        xu_2_3x3_conv2 = self._sep_conv2d_with_bn_relu(input_tensor=xu_2_3x3_add_mask, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name + "_xu_3x3_conv2")
        rx_2_3x3_conv2 = self._sep_conv2d_with_bn_relu(input_tensor=rx_2_3x3_add_mask, num_filters=num_filters,
                                                 kernel_size=[3, 3], training=training,
                                                 name=name + "_rx_3x3_conv2")

        x_2_3x3_predict = tf.layers.conv2d(inputs= x_2_3x3_conv2, filters=1, kernel_size=[1, 1],
                                           padding="SAME",
                                           strides=(1, 1),
                                           activation=tf.nn.tanh,
                                           kernel_initializer=tf.variance_scaling_initializer(),
                                           use_bias=True,
                                           name=name+"x_3x3_predict")
        xu_2_3x3_predict = tf.layers.conv2d(inputs= xu_2_3x3_conv2, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"xu_3x3_predict")
        rx_2_3x3_predict = tf.layers.conv2d(inputs=rx_2_3x3_conv2, filters=1, kernel_size=[1, 1],
                                            padding="SAME",
                                            strides=(1, 1),
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            use_bias=True,
                                            name=name+"rx_3x3_predict")

        # stage2_predict_concat = tf.concat([x_2_predict_drop, xu_2_predict_drop, rx_2_predict_drop], axis=-1,
        #                                   name=name + "_predict_concat")
        #
        # stage2_predict = tf.reduce_mean(stage2_predict_concat, axis=-1, name=name+"_predict_reduce_mean", keepdims=True)



        return x_2_3x3_predict, xu_2_3x3_predict, rx_2_3x3_predict

    def Concat_reduce(self, input1, input2, name):
        concat =  tf.concat([input1, input2], axis = -1, name=name +"_concat")
        concat_reduce = tf.reduce_mean(concat, axis=-1, name=name + "_concat_reduce", keepdims=True)

        return concat_reduce

    # def add_predicts(self, predicts_list, name):
    #     for i in range(len(predicts_list)):
    #         if i ==0:
    #             output_predict = predicts_list[0]
    #         else:
    #             output_predict = tf.add(predicts_list[i], output_predict, name=name+ str(i) + "_add_" + str(i-1))
    #
    #     return output_predict

    def final_FP_gnerator_dynamic_length(self, down_top_output_features, final_num_filter_FP, training):
        """

        :param down_top_output_features: from C1 to last stage C5  should have /2 dimension
        :param num_filter_FP: Size of the top-down layers used to build the feature pyramid , in paper 256, final_num_filter_F IS 512
                                1X1 CONVS HAS 512/2 =256 filters , each stage num_feature maps =  512/4
        :return: /2 feature maps concatation
        """
        print("FPN APPLYING!......................................")
        C_1x1_conv_FP_list = []
        final_P_list = []
        P_128_bn_relu_list = []
        P = None
        Final_Pyramid_feature = None
        with tf.variable_scope("Final_FP_gnerator"):
            # genrate 256 feature maps

            for i in range(len(down_top_output_features)):  # LEN =5 i=0 i=1,... i = 4
                t = len(down_top_output_features) - i - 1  # t=4 , t= 3 ,..., t = 0
                C_1x1_256 = tf.layers.conv2d(inputs=down_top_output_features[t],  # from the last feature
                                             filters=final_num_filter_FP // 2,
                                             kernel_size=[1, 1],
                                             padding="same",
                                             activation=None,
                                             kernel_initializer=tf.variance_scaling_initializer(),
                                             name="fpn_1x1_C" + str(
                                                 t + 1) + "_256")  # kernel_initializer: An initializer for the convolution kernel.
                C_1x1_conv_FP_list.insert(0, C_1x1_256)
            print(len(C_1x1_conv_FP_list))

            for i in range(len(C_1x1_conv_FP_list)):
                t = len(down_top_output_features) - i - 1
                if i == 0:
                    P = C_1x1_conv_FP_list[t]
                    final_P_list.append(P)
                elif i == len(C_1x1_conv_FP_list) - 1:  # i =1, t= 3
                    pass
                else:
                    UpS_P = tf.layers.conv2d_transpose(inputs=final_P_list[0], filters=final_num_filter_FP // 2,
                                                       kernel_size=[2, 2],
                                                       strides=(2, 2),
                                                       kernel_initializer=tf.variance_scaling_initializer(),
                                                       name='UpSx2_P' + str(t + 1))
                    P = tf.add(UpS_P, C_1x1_conv_FP_list[t], name="p_add")
                    final_P_list.insert(0, P)

                if i != len(C_1x1_conv_FP_list) - 1:
                    # generate 128 feature maps
                    P_128 = tf.layers.conv2d(inputs=P,  # use input P5
                                             filters=final_num_filter_FP // 4,
                                             kernel_size=[3, 3],
                                             padding="same",
                                             activation=None,
                                             kernel_initializer=tf.variance_scaling_initializer(),
                                             name="P" + str(
                                                 t + 1) + "_128")  # kernel_initializer: An initializer for the convolution kernel.
                    # P_128_bn = tf.layers.batch_normalization(P_128, training=True,
                    #                                          name="P" + str(t + 1) + "_128_bn")  # batch normalization

                    P_128_bn = batchnorm_with_name(P_128, training=training, name="P" + str(t + 1) + "_128_bn")
                    P_128_bn_relu = tf.nn.relu(P_128_bn, name="P" + str(t + 1) + "_128_bn_relu")
                    print("P_128_bn_relu---->", P_128_bn_relu)
                    P_128_bn_relu_list.insert(0, P_128_bn_relu)
            print("------------------------------------------------")
            # print(P_128_bn_relu)
            # # upsampling to P2 dimension to fusion
            # get the shape of P2
            print(P_128_bn_relu_list)
            target_shape = P_128_bn_relu_list[0].shape.as_list()[
                           1:3]  # [1:3]==> 表示[1,2] 不包括3；即：h = shape[0], w= shape[1]
            print("target_shape", target_shape)
            for i in range(len(P_128_bn_relu_list)):
                print(i)

                if i == 0:
                    Final_Pyramid_feature = P_128_bn_relu_list[i]
                else:

                    P_128_bn_relu_upsampling = tf.image.resize_bilinear(P_128_bn_relu_list[i], size=target_shape,
                                                                        align_corners=True,
                                                                        name="P" + str(
                                                                            i + 1) + "_blinear_128_bn_relu_x" + str(
                                                                            2 ** i))
                    Final_Pyramid_feature = tf.concat([Final_Pyramid_feature, P_128_bn_relu_upsampling],
                                                      axis=-1,
                                                      name="Final_Pyramid_feature_512")
            return Final_Pyramid_feature

    def new_final_FP_gnerator_dynamic_length(self, down_top_output_features, final_num_filter_FP, dimension_rate,training,name):
        """
         concate input feature list with specified final num of filters

        :param down_top_output_features: from C1 to last stage C5
        :param dimension_rate: such as 2, 4, 8, 16;  2 means half dimension. raw shape//2
        :param num_filter_FP:
        :return: /2 feature maps concatation
        """
        print("FPN APPLYING!......................................")
        final_feature_dim = [self.raw_shape[0]//dimension_rate, self.raw_shape[1]//dimension_rate]
        len_input_list = len(down_top_output_features)
        each_num_filters = final_num_filter_FP//len_input_list
        final_feature_concat = None
        for i in range(len_input_list):

            down_top_output_feature_resized = tf.image.resize_bilinear(down_top_output_features[i],
                                                                       final_feature_dim,
                                                             align_corners=True, name=name+"_feature_resized_"+str(i))
            # 1x1
            conv_1x1 = tf.layers.conv2d(inputs=down_top_output_feature_resized,
                                        filters=each_num_filters,
                                        kernel_size=[1, 1],
                                        padding="same",
                                        kernel_initializer=tf.initializers.truncated_normal(),
                                        name=name + "_conv_1x1_"+ str(i))
            conv_1x1_bn = batchnorm_with_name(conv_1x1, training=training, name= name + "_bn_"+ str(i))

            if i==0:
                final_feature_concat= conv_1x1_bn
            else:
                final_feature_concat = tf.concat([final_feature_concat, conv_1x1_bn], axis=-1)
        return final_feature_concat

    def conv_pool(self, input, filters_1, filters_2, kernel_size, training, name='conv2d_with_pool'):
        """
        this function is only use in U-NET
        :param input: input images
        :param filters_1: the number of kernels for the 1st conv-layer
        :param filters_2: the number of kernels for the 2nd conv-layer
        :param kernel_size: the size of kernel
        :param name: function name will use in tensorboard maybe
        :return:
        """
        with tf.variable_scope(name):
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            regularizer = None
            conv_1 = tf.layers.conv2d(inputs=input, filters=filters_1, kernel_size=kernel_size, padding="same",
                                      activation=None, kernel_regularizer=regularizer,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      name="conv_1")  # kernel_initializer: An initializer for the convolution kernel.
            # print(conv_1)
            conv1_batch_norm = tf.layers.batch_normalization(conv_1, axis=3, epsilon=1e-5, momentum=0.1,
                                                             training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02))  # batch normalization
            conv_1_activation = tf.nn.relu(conv1_batch_norm)  # activation
            # add drop out
            # conv_1_activation = tf.layers.dropout(inputs=conv_1_activation, rate=0.1)
            conv_2 = tf.layers.conv2d(inputs=conv_1_activation, filters=filters_2, kernel_size=kernel_size,
                                      padding="same",
                                      activation=None, kernel_regularizer=regularizer,
                                      kernel_initializer=tf.variance_scaling_initializer(), name="conv_2")
            # print(conv_2)
            conv2_batch_norm = tf.layers.batch_normalization(conv_2, axis=3, epsilon=1e-5, momentum=0.1,
                                                             training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02))  # batch normalization
            conv_2_activation = tf.nn.relu(conv2_batch_norm)  # activation
            # add drop out
            # conv_2_activation = tf.layers.dropout(inputs=conv_2_activation, rate=0.1)
            pool = tf.layers.max_pooling2d(inputs=conv_2_activation, pool_size=[2, 2], strides=2, padding="same",
                                           name='pool')
            return conv_2_activation, pool

    def upconv_concat(self, inputA, inputB, filters, kernel_size, training, name="upconv"):
        """
        :param inputA:  the input feature map
        :param inputB:  last same dimension conv ouput
        :param filters:
        :param kernel_size:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            regularizer = None
            up_conv = tf.layers.conv2d_transpose(inputs=inputA, filters=filters, kernel_size=kernel_size,
                                                 strides=(2, 2),
                                                 padding="same",
                                                 activation=None, kernel_regularizer=regularizer,
                                                 kernel_initializer=tf.variance_scaling_initializer(),
                                                 name='up_conv')
            # print(up_conv)
            up_conv_batch_norm = tf.layers.batch_normalization(up_conv, axis=3, epsilon=1e-5, momentum=0.1,
                                                               training=training,
                                                               gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                              0.02))  # batch normalization
            up_conv_activation = tf.nn.relu(up_conv_batch_norm)
            # copy and crop inputB as the same size as up_conv_activation
            # target_shape = tf.shape(up_conv_activation)
            # # print(target_shape[0], target_shape[1], target_shape[2], target_shape[3])
            # resized_image = tf.image.resize_image_with_crop_or_pad(up_conv_activation, target_shape[1], target_shape[2])
            return tf.concat([up_conv_activation, inputB], axis=-1,
                             name="concat")  # Concatenates tensors along one dimension

    def conv_without_pool(self, input, filters_1, filters_2, kernel_size, training, name='conv2d_without_pool'):
        """
        this function is only use in U-NET
        :param input: input images
        :param filters_1: the number of kernels for the 1st conv-layer
        :param filters_2: the number of kernels for the 2nd conv-layer
        :param kernel_size: the size of kernel
        :param name: function name will use in tensorboard maybe
        :return:
        """
        with tf.variable_scope(name):
            # conv2d 1
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            regularizer = None
            conv_1 = tf.layers.conv2d(inputs=input, filters=filters_1, kernel_size=kernel_size, padding="same",
                                      activation=None, kernel_regularizer=regularizer,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      name="conv_1")  # kernel_initializer: An initializer for the convolution kernel.

            conv1_batch_norm = tf.layers.batch_normalization(conv_1, axis=3, epsilon=1e-5, momentum=0.1,
                                                             training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02))  # batch normalization
            conv_1_activation = tf.nn.relu(conv1_batch_norm)  # activation
            # conv2d 2
            conv_2 = tf.layers.conv2d(inputs=conv_1_activation, filters=filters_2, kernel_size=kernel_size,
                                      padding="same",
                                      activation=None, kernel_regularizer=regularizer,
                                      kernel_initializer=tf.variance_scaling_initializer(), name="conv_2")

            conv2_batch_norm = tf.layers.batch_normalization(conv_2, axis=3, epsilon=1e-5, momentum=0.1,
                                                             training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02))  # batch normalization
            conv_2_activation = tf.nn.relu(conv2_batch_norm)  # activation

            return conv_2_activation

    def conv_bn_relu(self, input, filters, kernel_size, training, name="conv_bn_relu"):
        with tf.variable_scope(name):
            conv_1 = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, padding="same",
                                      activation=None, kernel_regularizer=None,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      name="conv_1")  # kernel_initializer: An initializer for the convolution kernel.

            conv1_batch_norm = tf.layers.batch_normalization(conv_1, axis=3, epsilon=1e-5, momentum=0.1,
                                                             training=training,
                                                             gamma_initializer=tf.random_normal_initializer(1.0,
                                                                                                            0.02))  # batch normalization
            conv_1_activation = tf.nn.relu(conv1_batch_norm)  # activation

            return conv_1_activation

    def convs_with_softmax(self, input, filters_1, filters_2, kernel_size, training, name='conv2d_with_softmax'):
        """
        this function is only use in U-NET
        :param input: input images
        :param filters_1: the number of kernels for the 1st conv-layer
        :param filters_2: the number of kernels for the 2nd conv-layer
        :param kernel_size: the size of kernel
        :param name: function name will use in tensorboard maybe
        :return:
        """
        with tf.variable_scope(name):
            # conv2d 1
            # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            regularizer = None
            conv_1_bn_relu = self.conv_bn_relu(input=input, filters=filters_1, kernel_size=kernel_size,
                                               name="conv_1_bn_relu", training=training)

            # conv2d 2
            conv_2_bn_relu = self.conv_bn_relu(input=conv_1_bn_relu, filters=filters_2, kernel_size=kernel_size,
                                               name="conv_2_bn_relu", training=training)

            # 1x1 conv
            output = tf.layers.conv2d(inputs=conv_2_bn_relu, filters=1, kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            name="output_image")
            # softmax
            output_shape = output.get_shape().as_list()
            print("outshape:", output_shape)
            # output_shape = tf.Print(output_shape,[output_shape],name + "shape:")
            output_reshape = tf.reshape(output, [self.batch_size, tf.multiply(output_shape[1], output_shape[2])])
            attention_soft_max = tf.nn.softmax(logits=output_reshape, axis=-1, name="softmax")
            # attention_soft_max = tf.Print(attention_soft_max, [attention_soft_max],"attention softmax:")
            softmax_sum =  tf.reduce_sum(attention_soft_max, axis=-1)
            softmax_sum = tf.Print(softmax_sum, [softmax_sum],"softmax sum:")
            tf.summary.scalar("softmax_sum1=", softmax_sum[0])
            tf.summary.scalar("softmax_sum1=", softmax_sum[1])
            output_reshape_back = tf.reshape(attention_soft_max, [self.batch_size, output_shape[1], output_shape[2], output_shape[3]], name="output_reshape_back")

            tf.summary.image("attention_soft_max", output_reshape_back, max_outputs=self.batch_size)

            return output_reshape_back

    def create_attention_generator(self, inputs, training):
        with tf.variable_scope("required_parameters"):
            conv_1, pool_1 = self.conv_pool(inputs, 64, 64, [3, 3], training=training, name="conv_pool_1")

            conv_2, pool_2 = self.conv_pool(pool_1, 128, 128, [3, 3], training=training, name="conv_pool_2")

            conv_3, pool_3 = self.conv_pool(pool_2, 256, 256, [3, 3], training=training, name="conv_pool_3")

            conv_4, pool_4 = self.conv_pool(pool_3, 512, 512, [3, 3], training=training, name="conv_pool_4")

            conv_5 = self.conv_without_pool(pool_4, 1024, 1024, [3, 3], training=training,
                                            name="conv_without_pool_5")

        print(conv_5)

        # dropout
        conv_5 = tf.Print(conv_5, [conv_5], "conv5 before Dropout: ")
        conv_5_dropout = tf.layers.dropout(inputs=conv_5, rate=0.5, training=training)
        conv_5_dropout = tf.Print(conv_5_dropout, [conv_5_dropout], "conv5 after Dropout: ")
        upconv_6 = self.upconv_concat(conv_5_dropout, conv_4, 512, [2, 2], training=training,
                                      name="upconv_6")  # 拼接以便 可以下一次还原到
        conv_6 = self.conv_without_pool(upconv_6, 512, 512, [3, 3], training=training,
                                        name="conv_without_pool_6")

        upconv_7 = self.upconv_concat(conv_6, conv_3, 256, [2, 2], training=training, name="upconv_7")
        C4_output = self.convs_with_softmax(upconv_7, 256, 128, [3, 3], training=training,
                                                    name="convs_with_softmax_7")

        return C4_output, conv_2, conv_1


    def final_output_Discriminator(self, C4_output, conv_2, conv_1, training):
            conv_7 = self.conv_bn_relu(input=C4_output, filters=128, kernel_size=[3, 3],
                                               name="conv_7_bn_relu", training=training)

            upconv_8 = self.upconv_concat(conv_7, conv_2, 128, [2, 2], training=training, name="upconv_8")
            conv_8 = self.conv_without_pool(upconv_8, 128, 128, [3, 3], training=training,
                                            name="conv_without_pool_8")

            upconv_9 = self.upconv_concat(conv_8, conv_1, 64, [2, 2], training=training, name="upconv_9")

            conv_9 = self.conv_without_pool(upconv_9, 64, 64, [3, 3], training=training, name="conv_without_pool_9")

            # use tanh
            output_image = tf.layers.conv2d(inputs=conv_9, filters=1, kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                            name="output_image")
            return output_image, conv_7, upconv_8, conv_8, upconv_9, conv_9

    def create_model(self, inputs, training):
        """
        only last two blocks used atrous convs with rate 2 and 4 for output stride 8（1/8 as input image）:
        :param inputs:
        :param training:
        :param filter_number:
        :return:
        """
        with tf.variable_scope(self.name) as scope:
            # some initialization ------------------>
            # training=tf.Print(training,[training],"Training switch value:")
            with tf.variable_scope("attention_generator"):
                C4_output, conv_2, conv_1 = self.create_attention_generator(inputs=inputs, training=training)
                c4_pred_resized = tf.image.resize_bilinear(C4_output, self.raw_shape,
                                                           align_corners=True, name="c4_pred_resized")
            with tf.variable_scope("final_out_seg"):
                output_image, \
                conv_7, upconv_8, conv_8, upconv_9, conv_9= self.final_output_Discriminator(
                    C4_output=C4_output, conv_2=conv_2, conv_1= conv_1, training=training)


            return  output_image, c4_pred_resized,\
                    conv_1, conv_2, conv_7, upconv_8, conv_8, upconv_9, conv_9




    @staticmethod
    def compute_sigmoid_cross_entropy_loss(logits, labels, name=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name=name)


    @staticmethod
    def compute_cross_entropy_loss(logits, labels, name=None):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name=name)

    @staticmethod
    def compute_negative_likelyhood_loss(logits, EPS, name=None):
        """
        :param logits: prediction values in range [0, 1]
        :param EPS:  A small value which used to avoid denominator =0
        :param name:
        :return:
        """
        return tf.reduce_mean(-tf.log(logits + EPS), name=name)

    @staticmethod
    def compute_L1_loss(logits, labels, name=None):
        """
        :param logits: prediction values in range [0, 1]
        :param EPS:  A small value which used to avoid denominator =0
        :param name:
        :return:
        """
        return tf.reduce_mean(tf.abs(labels - logits), name=name)

    def view_feature(self, batch_features, name):
        """
        :param batch_features:  shape=[batch_size, w, h , # of channels]
        :return:
        """
        feature_shape = batch_features.get_shape().as_list()
        batch_features =tf.reshape(batch_features,(self.batch_size, feature_shape[1], feature_shape[2], feature_shape[3]))
        each_input_feature_list = tf.unstack(batch_features, axis=0)
        print(each_input_feature_list)
        for i in range(self.batch_size):
            with tf.name_scope(str(i)):
            # # delete batch size dimension squeeze: delte the dimension with "1"
            # each_feature_maps = tf.squeeze(each, axis=0)
                each_feature_maps_tranposed = tf.transpose(tf.expand_dims(each_input_feature_list[i], axis=0), (3, 1, 2, 0))
                tf.summary.image(name, each_feature_maps_tranposed, max_outputs=feature_shape[-1])

    def evaluate_model(self, input_img, training=True,):
        """
        :param input_data: placehold for input image
        :param data:  data[0] input_image data[1] batch_label data
        :param reuse:
        :param training:
        :param suffix:
        :return:
        """

        # create model：
        with tf.name_scope("model_outputs"):
            output_image, self.c4_pred_resized, \
            conv_1, conv_2, \
            conv_7, upconv_8, conv_8, upconv_9, conv_9= self.create_model(input_img, training=training)

            tf.summary.image("c4_pred_resized", self.c4_pred_resized, max_outputs=self.batch_size)
            tf.summary.image("output_image", output_image, max_outputs=self.batch_size)


        # # view each feature
        # with tf.name_scope("each_conv_5_dropout"):
        #     self.view_feature(conv_5_dropout, "conv_5_dropout")
        #
        #
        # # view mean features
        with tf.name_scope("conv_1"):
            conv_1 = tf.reduce_mean(conv_1, axis=-1, keepdims=True, name="conv_1")
            tf.summary.image("conv_1", conv_1, max_outputs=self.batch_size)

        with tf.name_scope("conv_2"):
            conv_2 = tf.reduce_mean(conv_2, axis=-1, keepdims=True, name="conv_2")
            tf.summary.image("conv_2", conv_2, max_outputs=self.batch_size)
        #
        # with tf.name_scope("conv_3"):
        #     conv_3 = tf.reduce_mean(conv_3, axis=-1, keepdims=True, name="conv_3")
        #     tf.summary.image("conv_3", conv_3, max_outputs=self.batch_size)
        #
        # with tf.name_scope("conv_4"):
        #     conv_4 = tf.reduce_mean(conv_4, axis=-1, keepdims=True, name="conv_4")
        #     tf.summary.image("conv_4", conv_4, max_outputs=self.batch_size)
        #
        # with tf.name_scope("conv_5"):
        #     conv_5 = tf.reduce_mean(conv_5, axis=-1, keepdims=True, name="conv_5")
        #     tf.summary.image("conv_5", conv_5, max_outputs=self.batch_size)
        #
        # with tf.name_scope("conv_5_dropout"):
        #     conv_5_dropout = tf.reduce_mean(conv_5_dropout, axis=-1, keepdims=True, name="conv_5_dropout")
        #     tf.summary.image("conv_5_dropout", conv_5_dropout, max_outputs=self.batch_size)
        #
        #
        # with tf.name_scope("upconv_6"):
        #     upconv_6 = tf.reduce_mean(upconv_6, axis=-1, keepdims=True, name="upconv_6")
        #     tf.summary.image("upconv_6", upconv_6, max_outputs=self.batch_size)
        #
        # with tf.name_scope("conv_6"):
        #     conv_6 = tf.reduce_mean(conv_6, axis=-1, keepdims=True, name="conv_6")
        #     tf.summary.image("conv_6", conv_6, max_outputs=self.batch_size)
        #
        # with tf.name_scope("upconv_7"):
        #     upconv_7 = tf.reduce_mean(upconv_7, axis=-1, keepdims=True, name="upconv_7")
        #     tf.summary.image("upconv_7", upconv_7, max_outputs=self.batch_size)
        #
        with tf.name_scope("conv_7"):
            conv_7 = tf.reduce_mean(conv_7, axis=-1, keepdims=True, name="conv_7")
            tf.summary.image("conv_7", conv_7, max_outputs=self.batch_size)
        #
        with tf.name_scope("upconv_8"):
            upconv_8 = tf.reduce_mean(upconv_8, axis=-1, keepdims=True, name="upconv_8")
            tf.summary.image("upconv_8", upconv_8, max_outputs=self.batch_size)

        with tf.name_scope("conv_8"):
            conv_8_mean = tf.reduce_mean(conv_8, axis=-1, keepdims=True, name="conv_8_mean")
            self.conv_8_mean_resized = tf.image.resize_bilinear(conv_8_mean, self.raw_shape,
                                                       align_corners=True, name="conv_8_mean_resized")
            tf.summary.image("conv_8_mean", conv_8_mean, max_outputs=self.batch_size)

        with tf.name_scope("upconv_9"):
            upconv_9 = tf.reduce_mean(upconv_9, axis=-1, keepdims=True, name="upconv_9")
            tf.summary.image("upconv_9", upconv_9, max_outputs=self.batch_size)

        with tf.name_scope("conv_9"):
            conv_9 = tf.reduce_mean(conv_9, axis=-1, keepdims=True, name="conv_9")
            tf.summary.image("conv_9", conv_9, max_outputs=self.batch_size)
        #




        with tf.name_scope("Losses"):
            zero_to_one_label = tf.divide(tf.add(self.vein_inputs_label, 1, name="softmax_label_add_1"), 2.0,
                                          name="zero_to_one_label")
            tf.summary.image("zero_to_one_label", zero_to_one_label, max_outputs=self.batch_size)
            # # flat output and intput
            # zero_to_one_label_flat =  tf.reshape(zero_to_one_label, [self.batch_size, self.raw_shape[0]*self.raw_shape[1]])
            #
            # zero_to_one_pred_flat = tf.reshape(output_image, [self.batch_size, self.raw_shape[0] * self.raw_shape[1]])
            # c4_pred_resized_flate = tf.reshape(c4_pred_resized, [self.batch_size, self.raw_shape[0] * self.raw_shape[1]])

            with tf.name_scope("Generator_Attention_Loss"):
                C4vein_entropy_losses = self.compute_sigmoid_cross_entropy_loss(
                    logits=self.c4_pred_resized, labels=zero_to_one_label, name="C4vein_entropy_losses")
                C4vein_entropy_losses = tf.Print(C4vein_entropy_losses, [C4vein_entropy_losses],
                                                 "C4vein_entropy_losses: ")
                C4vein_entropy_losses = C4vein_entropy_losses * self.output_entropy_weight

                C4vein_entropy_losses_sum = tf.summary.scalar("C4vein_entropy_losses", C4vein_entropy_losses)

            with tf.name_scope("Final_output_Loss"):

                # final_vein_entropy_losses_ESC = self.compute_sigmoid_cross_entropy_loss(
                #     logits=output_image, labels=self.vein_inputs_label, name="final_vein_entropy_losses_ESC")
                # final_vein_entropy_losses_ESC = tf.Print(final_vein_entropy_losses_ESC, [final_vein_entropy_losses_ESC], "final_vein_entropy_losses_ESC: ")
                # final_vein_entropy_losses_ESC = final_vein_entropy_losses_ESC * self.output_entropy_weight

                final_vein_loss_L1 = tf.reduce_mean(tf.abs(self.vein_inputs_label - output_image))
                final_vein_loss_L1 = final_vein_loss_L1 * self.output_l1_weight
                final_vein_loss_L1 = tf.Print(final_vein_loss_L1, [final_vein_loss_L1], "final_vein_loss_L1: ")
                vein_L1_losses_sum = tf.summary.scalar("final_vein_loss_L1", final_vein_loss_L1)



                final_vein_losses =  final_vein_loss_L1
                final_vein_losses_sum = tf.summary.scalar("final_vein_losses", final_vein_losses)



        return output_image, C4vein_entropy_losses, final_vein_losses



    def build_model(self):
        """
        This function is as "train_process" from old framework
        :return:
        """
        # # # Define placeholders for input values ( not labels)
        self.inputs_data = tf.placeholder(tf.float32, [None, self.raw_shape[0], self.raw_shape[1], 3],
                                          name='input_image')
        tf.summary.image("inputs_image", self.inputs_data, max_outputs=self.batch_size)

        self.tongue_inputs_label = tf.placeholder(tf.float32, [None, self.raw_shape[0], self.raw_shape[1], 1],
                                           name='inputs_tongue_label')
        tf.summary.image("inputs_tongue_label", self.tongue_inputs_label, max_outputs=self.batch_size)
        #
        self.vein_inputs_label = tf.placeholder(tf.float32, [None, self.raw_shape[0], self.raw_shape[1], 1],
                                                  name='inputs_vein_label')
        tf.summary.image("inputs_vein_label", self.vein_inputs_label, max_outputs=self.batch_size)
        #------------------------------------

        self.training_HD = tf.placeholder(tf.bool, name= "training_parameters_switch")
        # Define placeholder for dataset handle ( to select training or validation)
        self.dataset_handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')

        # Get handles from string handle
        self.iterator = tf.data.Iterator.from_string_handle(self.dataset_handle,
                                                            self.train_batch_data_iterator.output_types,
                                                            self.train_batch_data_iterator.output_shapes)
        self.data = self.iterator.get_next(name="input_batch_data")
        print("self.data------------------:", self.data)

        # Using Poly learning rate policy
        # base_lr = tf.constant( self.learning_rate)
        # step_ph = tf.placeholder( dtype=tf.float32, shape=( ))
        # self.training_hd = tf.placeholder(tf.bool, name="training_hd")



        # self.learning_rate = tf.train.exponential_decay(self.learning_rt, self.global_step,
        #                                                 self.lr_decay_step, self.lr_decay_rate)



        # evaluation
        self.pred, self.C4vein_entropy_losses, self.final_vein_entropy_losses=\
        self.evaluate_model(input_img=self.inputs_data,
                            training=self.training_HD)

        all_trainable = [v for v in tf.trainable_variables() if
                         ('beta' not in v.name and 'gamma' not in v.name)]
        # get weights variables
        weights = [v for v in all_trainable if 'kernel' in v.name]  # lr * 10.0

        # get weight variables
        print(all_trainable)
        print(weights)
        d_vars = [var for var in tf.trainable_variables() if 'final_out_seg' in var.name]
        g_vars = [var for var in tf.trainable_variables() if 'attention_generator' in var.name]

        print("d_vars length：", len(d_vars))
        print("g_vars length：", len(g_vars))

        # Define optimizers for training the discriminator and generator

        # Define optimizers for training the discriminator and generator
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.name_scope("Generator_Optimizer"):
                    self.g_optim = tf.train.AdamOptimizer(self.learning_rt * 5, beta1=self.adam_beta1) \
                        .minimize(self.C4vein_entropy_losses, var_list=g_vars)
            with tf.name_scope("Discriminator_Optimizer"):
                with tf.control_dependencies([self.g_optim]):
                    self.d_optim = tf.train.AdamOptimizer(self.learning_rt, beta1=self.adam_beta1) \
                        .minimize(self.final_vein_entropy_losses, var_list=d_vars, global_step=self.global_step)



        self.total_loss = self.final_vein_entropy_losses + self.C4vein_entropy_losses
        total_loss_sum = tf.summary.scalar("total_loss", self.total_loss)

        # prediction transform
        self.prediction = tf.divide(tf.add(self.pred, 1, name="prediction_add_1"), 2.0, name="predict_divide")
        self.prediction = tf.identity(self.prediction, name="prediction")
        self.merged_summaries = tf.summary.merge_all()

        #Evaluation
        with tf.name_scope("Evaluation"):
            with tf.name_scope("Confusion_Matrix"):
                # # flatten prediction and label
                self.label_zero_to_one = tf.divide(tf.add(self.vein_inputs_label, 1, name="label_add_1"), 2.0,
                                                   name="label_0_to_1")
                # self.label_zero_to_one = tf.divide(tf.add(self.tongue_inputs_label, 1, name="label_add_1"), 2.0,
                #                                    name="label_0_to_1")

                self.threshold = tf.placeholder(tf.float32, shape=[], name='threshold')  # placeholder for binarization

                self.th_mask = tf.image.resize_bilinear(tf.reshape(self.threshold, [-1, 1, 1, 1]), self.raw_shape,
                                                        align_corners=True, name="th_mask")
                self.predict_th_substract = tf.subtract(
                    self.prediction,
                    self.th_mask,
                    name="Subtract_prediction_with_th"
                )
                # use substraction result to binarize the tongue prediction
                self.bi_prediction = tf.nn.relu(tf.sign(self.predict_th_substract),
                                                name="binarized_prediction")
                self.th_summary = tf.summary.image("binarized_prediction", self.bi_prediction,
                                                   max_outputs=self.batch_size)
                self.TP = tf.count_nonzero(tf.multiply(self.bi_prediction, self.label_zero_to_one))
                self.FP = tf.count_nonzero(tf.multiply(self.bi_prediction, tf.subtract(self.label_zero_to_one, 1)))
                self.FN = tf.count_nonzero(tf.multiply(tf.subtract(self.bi_prediction, 1), self.label_zero_to_one))
                self.TN = tf.count_nonzero(
                    tf.multiply(tf.subtract(self.bi_prediction, 1), tf.subtract(self.label_zero_to_one, 1)))

                # iou_tp = TP / ( TP + FP + FN)
                self.IoU = tf.divide(self.TP, tf.add(tf.add(self.TP, self.FP), self.FN))


            with tf.name_scope("Scaler_Writer"):
                # self.writer_name = tf.placeholder(tf.string, shape=[], name='writer_name')

                self.IoU_value = tf.placeholder(tf.float32, shape=[], name='scaler_value')  # placeholder for binarization
                self.IoU_sum = tf.summary.scalar("IoU",  self.IoU_value)

        self.saver = tf.train.Saver()

    def load_weights(self, data_path, sess, required_var_scope):
        m_name= self.model_name
        print(required_var_scope)
        data_dict = np.load(data_path, encoding='latin1').item()
        keys = sorted(data_dict.keys())

        print("saved keys:------------------>",keys)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print(vars_list)
        count=0
        for each in vars_list:
            print("current graph variable names:", each.name)
            if required_var_scope in each.name:
                print("extract varaible name:", each.name)
                # for i in keys:
                #     if i in each.name:
                #         print("related load key name:", i)
                # "copy weights to the sess variable with contains "required_var_scope"
                # assert key in vars_list.keys(), "variable name is not matched"
                key= each.name.rstrip(":0").lstrip(m_name)
                print("final_key:", key)
                if key in keys:
                    before = each.eval()
                    print("before assign:", before)
                    sess.run(each.assign(data_dict[key]))
                    after = each.eval()
                    print("after assign:", after)
                    print("data_dict[key]：", data_dict[key])

                    count+=1
                else:
                    print(key, "------------------------------skipped--!")
            else:
                print("something wrong")
                print(each.name, "-------------------------------------skip!----------")
        print("...............................................")
        print("graph variable length=:", len(vars_list))
        print("assigned variable length=:", count)

        # save one more time model for later check whether the variables correctly restored



    def train(self):
        # Define summary writer for saving log files ( for training and validation)
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'training/'), graph=tf.get_default_graph())
        self.vwriter = tf.summary.FileWriter(os.path.join(self.log_dir, 'validation/'), graph=tf.get_default_graph())
        self.IoU_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'vIoU/'), graph=tf.get_default_graph())
        #

        # self.export_dir= "./Model/Simple_save/"
        # Show list of all variables and total parameter count
        show_variables()
        print("\n[ Initializing Variables ]\n")

        # childs( self.prediction)
        # print( "\n[ Check prediction property ]\n")
        # show_op_names( self)
        #
        # show_tensor_names( self)

        # ######################### load weights #########################
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        # self.load_weights('New_pretrained_desired_partial_weights.npy', self.sess, "required_parameters")


        # Get handles for training and validation datasets // push the iterator handle is like to pop up the data flow
        self.train_handle, self.validation_handle = self.sess.run([self.train_batch_data_iterator.string_handle(),
                                                                   self.val_batch_data_iterator.string_handle()])

        # some initial parameters for training
        # self.avg_v_target_loss_per_epoch=100.0
        self.avg_IoU_per_epoch = 0.0
        # self.avg_IoU_per_epoch = 0.0
        self.target_avg_IoU_per_epoch = 0.0
        data_count = 0
        # self.temp_per_epoch_loss_list = []
        self.temp_per_epoch_IoU_list = []
        self.model_saved_log = {}
        self.model_saved_log["Total_num_save"] = 0
        self.valIoU_track_global = []

        self.g_loss_list = []
        self.d_loss_list = []
        self.total_loss_list = []
        self.track_saved_all = [] #  inside contains all losses lists -> 0: validate IOu ,1: g_loss_list, 2: d_loss_list 3: total_loss_list
        th = np.linspace(0.05, 0.95, 10)

        # Iterate through training steps
        # while not self.sess.should_stop( ):




        while True:
            try:
                # Update global step
                # step = tf.train.global_step(self.sess, self.global_step)
                # print(step)
                # set up training switch=  True
                self.training = True

                # self.drop_Matrix = self.generateAllTrueArray([4, 3])

                # try to extract data from records datapiple line
                data_fd = {self.dataset_handle: self.train_handle}  # dataset_handle is a placeholder of handle
                # print("!!!!!train_handle", self.train_handle)
                # data_fd = {self.dataset_handle: self.validata_handle}  # dataset_handle is a placeholder of handle
                data = self.sess.run(self.data, feed_dict=data_fd)
                real_inputs_fd = {self.inputs_data: data[0], self.vein_inputs_label: data[1],
                                  self.tongue_inputs_label: data[2], self.training_HD: self.training}
                # real_inputs_fd = {self.inputs_data: data, self.training_HD: self.training}


                # start to train
                # train and print result
                step = tf.train.global_step(self.sess, self.global_step)
                self.step =step
                self.train_total_loss, self.g_loss, self.d_loss,_ ,_= self.sess.run(
                    [self.total_loss, self.C4vein_entropy_losses, self.final_vein_entropy_losses, self.d_optim,
                     self.g_optim],
                        feed_dict=real_inputs_fd)
                self.g_loss_list.append(self.g_loss)
                self.d_loss_list.append(self.d_loss)
                self.total_loss_list.append(self.train_total_loss)


                # check whether needs to write logs
                if (step % self.summary_step == 0):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = self.sess.run(self.merged_summaries, feed_dict=real_inputs_fd, options=run_options,
                                            run_metadata=run_metadata)
                    print("writing training log ....")
                    self.writer.add_run_metadata(run_metadata, tag="step%d" % step)
                    self.writer.add_summary(summary, step)
                    self.writer.flush()
                    print("log is done")

                # validating --------------------
                self.training = False
                data_fd = {self.dataset_handle: self.validation_handle}  # dataset_handle is a placeholder of handle
                data = self.sess.run(self.data, feed_dict=data_fd)

                self.real_vinputs_fd = {self.inputs_data: data[0], self.vein_inputs_label: data[1],
                                        self.tongue_inputs_label: data[2], self.training_HD: self.training}
                # self.real_vinputs_fd= {self.inputs_data: data, self.training_HD: self.training}


                # - set up training switch = False:



                vsummary, self.val_total_loss = \
                    self.sess.run([self.merged_summaries, self.total_loss],
                                  feed_dict=self.real_vinputs_fd)

                # calculate IoU
                self.max_IOU_TP, self.mean_IOU_TP, self.opt_th, _ = self.custom_metric_IOU_TP(
                    real_fd=self.real_vinputs_fd, th=th, th_summary_bool=True)

                self.temp_per_epoch_IoU_list.append(self.mean_IOU_TP)

                if step % self.num_batches_per_epoch == 0:
                    self.avg_IoU_per_epoch = np.mean(self.temp_per_epoch_IoU_list)

                    if self.avg_IoU_per_epoch > self.target_avg_IoU_per_epoch:
                        # self.training = False
                        # self.drop_Matrix = self.generateAllTrueArray(shape=[4, 3])
                        # write log
                        print("writing validation log ....")
                        self.vwriter.add_summary(vsummary, step)
                        self.vwriter.flush()
                        print("log is done.")
                        # save_model
                        print("normal model saving ------------------------------------>")

                        self.saver.save(self.sess, self.checkpoint_dir + "normal_model.ckpt")
                        # print( "writing .pbtxt")
                        tf.train.write_graph(self.sess.graph.as_graph_def(), self.checkpoint_dir,
                                             'graph_node.pbtxt', as_text=True)
                        self.model_saved_log["step"] = self.step
                        self.model_saved_log["Total_num_save"] = self.model_saved_log["Total_num_save"] + 1
                        print("model saved done--------------------------------->")

                        # replace target loss
                        # self.avg_v_target_loss = avg_v_total_loss
                        self.target_avg_IoU_per_epoch = self.avg_IoU_per_epoch

                        self.temp_per_epoch_IoU_list = []
                        # freeze graph
                        # self.sess.graph.finalize( )

                        # Plot predictions , fake and real
                        print("saving plotting")
                        self.plot_predictions(step, real_fd=self.real_vinputs_fd)
                        # self.plot_prediction_from_saver( )
                        # self.plot_image( self.predict_fake, step, self.real_vinputs_fd, name="predict_fake")
                        # self.plot_image( self.predict_real, step, self.real_vinputs_fd, name="predict_real")

                        # write IoU into the tensorboard
                        self.IoU_scaler_fd = {self.IoU_value: self.target_avg_IoU_per_epoch}
                        IoUsummary = self.sess.run(self.IoU_sum, feed_dict=self.IoU_scaler_fd)
                        self.IoU_writer.add_summary(IoUsummary, step)
                        self.IoU_writer.flush()
                    else:
                        pass

                # print performance:
                print("--------------------------------->")
                print(
                    "[%s / %s]: \n train-> %.10f [total loss] ;"
                    "\n validate->%.10f [total_loss]; \n val_max_IoU->%.10f( opt:%0.3f); val_mean_IoU->%.10f;"
                    "\n current_per_epoch_IoU:%.10f"
                    % (step + 1, self.num_batches,
                       self.train_total_loss, self.val_total_loss,
                       self.max_IOU_TP, self.opt_th,
                       self.mean_IOU_TP,
                       self.target_avg_IoU_per_epoch))



                self.valIoU_track_global.append(self.mean_IOU_TP)







            except tf.errors.OutOfRangeError:
                # plt.ioff( )
                # print( test_batches_image_souls.shape)
                # print( test_batches_label_souls.shape)
                # print( "labels:range")
                # print("writing IoU tracker into disk")
                # storeTree(self.valIoU_track_global, self.checkpoint_dir, self.name + "_valIoU_list_data")
                self.track_saved_all.append(self.valIoU_track_global)
                self.track_saved_all.append(self.g_loss_list)
                self.track_saved_all.append(self.d_loss_list)
                self.track_saved_all.append(self.total_loss_list)
                print("Saved tracking IoU len:", len(self.track_saved_all[0]))
                print("Saved tracking g_loss_list len:", len(self.track_saved_all[1]))
                print("Saved tracking d_loss_list len:", len(self.track_saved_all[2]))
                print("Saved tracking total_loss_list len:", len(self.track_saved_all[3]))
                storeTree(self.track_saved_all, self.IoU_tracking_saved_dir, self.name + "_track_saved_all")
                print("data save finished.!:D")
                print("In total, model saved %s times." % self.model_saved_log["Total_num_save"])
                print("The last save is at step %s." % self.model_saved_log["step"])


                break

    # model functions
    def predict(self, real_fd):

        # set up training
        # self.training = False
        return self.sess.run(self.prediction, feed_dict=real_fd)

        # Plot generated images for qualitative evaluation

    def plot_predictions(self, step, real_fd):
        # plot_subdir = os.path.join( self.plot_dir, str( step))
        plot_subdir = os.path.join(self.plot_dir)
        checkFolders([self.plot_dir, plot_subdir])
        resized_imgs = self.predict(real_fd=real_fd)
        print("validate prediction shape:", resized_imgs.shape)
        for n in range(0, resized_imgs.shape[0]):
            plot_name = 'val_plot_step_' + str(step) + "_img_" + str(n) + '.png'
            plt.imsave(os.path.join(plot_subdir, plot_name), resized_imgs[n, :, :, 0], cmap='gray')

    def plot_prediction_from_saver(self, real_fd):
        self.saver.restore(self.sess, self.checkpoint_dir + "normal_model.ckpt")
        predict_img = self.predict(real_fd=real_fd)
        print("predict shape", predict_img.shape)
        predict_temp = predict_img[0, :, :, :].reshape((predict_img.shape[1], predict_img.shape[2]))
        plt.imshow(predict_temp, cmap="gray")
        plt.show()
        plt.pause(0.005)
        # Compute cumulative loss over multiple batches

    def plot_image(self, image, step, real_fd, name="val"):
        """
        :param image: the tensor needs to be evaluated and saved in disk
        :param step:  global step
        :return:
        """
        # plot_subdir = os.path.join( self.plot_dir, str( step))
        self.training = False
        plot_subdir = os.path.join(self.plot_dir)
        checkFolders([self.plot_dir, plot_subdir])
        resized_imgs = self.sess.run(image, feed_dict=real_fd)
        print("validate prediction shape:", resized_imgs.shape)
        for n in range(0, resized_imgs.shape[0]):
            plot_name = name + '_val_plot_step_' + str(step) + "_img_" + str(n) + '.png'
            plt.imsave(os.path.join(plot_subdir, plot_name), resized_imgs[n, :, :, 0], cmap='gray')

    def compute_cumulative_loss(self, loss, loss_ops, dataset_handle, batches):
        self.training = False
        for n in range(0, batches):
            fd = {self.dataset_handle: dataset_handle}
            current_loss = self.sess.run(loss_ops, feed_dict=fd)
            loss = npx = tf.add(loss, current_loss)
            sys.stdout.write('Batch {0} of {1}\r'.format(n + 1, batches))
            sys.stdout.flush()
        return loss

    def custom_metric_IOU_TP(self, real_fd, th, th_summary_bool=False):
        """
        :param test_input:  the input tensor for prediction
        :param test_labels:  the labels tensorfor the output : true values
        :param th : the list of thresholds which used to threshold predictions
        :return: max_IOU_TP, max_index, opt_th
        """

        # # print( "predict shape", predicts.shape)
        # p_min = np.min( predict_batch)
        # p_max = np.max( predict_batch)
        # print( "prediction range[%0.3f， %0.3f]" % ( p_min, p_max))
        # label_batch_sk = label_batch.astype( int)
        # # print( img_batch.shape)
        # # print( predicts.shape)
        # predicts_sk = predict_batch.flatten( ).reshape( 
        #     predict_batch.shape[0] * predict_batch.shape[1] * predict_batch.shape[2])
        #
        # labels_sk = label_batch_sk.flatten( ).reshape( 
        #     predict_batch.shape[0] * predict_batch.shape[1] * predict_batch.shape[2])
        IOU_list_TP = []

        for eacht in th:
            real_fd[self.threshold] = eacht
            # print( real_fd)
            self.actual_iou = self.sess.run(self.IoU, feed_dict=real_fd)

            # print( self.actual_iou)
            IOU_list_TP.append(self.actual_iou)
            # print( eacht)
        max_IOU_TP = np.max(IOU_list_TP)
        # print( max_IOU_TP)
        mean_IOU_TP = np.mean(IOU_list_TP)
        # print( mean_IOU_TP)
        # print( "max_IOU_TP", max_IOU_TP)
        max_index = np.argmax(IOU_list_TP)
        opt_th = th[max_index]

        if th_summary_bool == True and self.step % (self.num_batches_per_epoch) == 0:
            # calculate from tensorflow
            real_fd[self.threshold] = opt_th

            th_summary, merged = self.sess.run([self.th_summary, self.merged_summaries], feed_dict=real_fd)
            self.vwriter.add_summary(th_summary, self.step)
            self.writer.flush()
            self.vwriter.add_summary(merged, self.step)
            self.writer.flush()
        else:
            pass
        return max_IOU_TP, mean_IOU_TP, opt_th, IOU_list_TP




# Initialize and train model
def main():
    # Define model parameters and options in dictionary of flags
    # FLAGS = getFlags_PSPNet( )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Initialize model
    model = Our_L1_SA_U_Net()

    # Specify number of training steps
    training_steps = model.num_batches
    print("total training step:", training_steps)

    with tf.Session() as sess:
        model.set_session(sess)

        # Train model
        model.train()

    print("\n[ TRAINING COMPLETE ]\n")


# Run main( ) function when called directly
if __name__ == '__main__':
    main()
