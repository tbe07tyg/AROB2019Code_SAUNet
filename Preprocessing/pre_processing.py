import tensorflow as tf
# data augmentation
from scipy import misc
import cv2



# <editor-fold desc="help function">
def rotate_image_func(image, angle):
    # 旋转角度: angle
    return misc.imrotate(image, angle, 'bicubic')
# </editor-fold>


# <editor-fold desc="flip">
def flip_left_right_dataset(image, label):
    fliped_image = tf.image.flip_left_right(image=image)
    fliped_label = tf.image.flip_left_right(image=label)
    return  fliped_image, fliped_label

def flip_left_right_dataset_tongue_and_vein(image, tongue_label, vein_label):
    fliped_image = tf.image.flip_left_right(image=image)
    fliped_tongue_label = tf.image.flip_left_right(image=tongue_label)
    fliped_vein_label = tf.image.flip_left_right(image=vein_label)
    return  fliped_image, fliped_tongue_label, fliped_vein_label


def flip_up_down_dataset(image, label):
    fliped_image = tf.image.flip_up_down(image=image)
    fliped_label = tf.image.flip_up_down(image=label)
    return  fliped_image, fliped_label

def flip_up_down_dataset_tongue_and_vein(image, tongue_label, vein_label):
    fliped_image = tf.image.flip_up_down(image=image)
    fliped_tongue_label = tf.image.flip_up_down(image=tongue_label)
    fliped_vein_label = tf.image.flip_up_down(image=vein_label)
    return  fliped_image, fliped_tongue_label, fliped_vein_label
# </editor-fold>


# <editor-fold desc="rotation">
def rotate_dataset_45(image, label):
    label_shape = tf.shape(label)
    label = tf.reshape(label, (label_shape[0], label_shape[1])) # for the rotate function, the input image either with shape(x,y,3) or shape(x,y)
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 45], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 45], tf.uint8), tf.float32)
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label


def rotate_dataset_135(image, label):
    label_shape = tf.shape(label)
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 135], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 135], tf.uint8), tf.float32)
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label


def rotate_dataset_225(image, label):
    label_shape = tf.shape(label)
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 225], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 225], tf.uint8), tf.float32)
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_10(image, label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    print("image shape:",image_shape)
    print("label shape:",label_shape)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 10], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 10], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 1))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_10_RGB(image, label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    print("image shape:",image_shape)
    print("label shape:",label_shape)
    image = tf.reshape(image, (image_shape[0], image_shape[1], 3))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 10], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 10], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 3))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_20_RGB(image, label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    print("image shape:",image_shape)
    print("label shape:",label_shape)
    image = tf.reshape(image, (image_shape[0], image_shape[1], 3))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 20], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 20], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 3))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_30_RGB(image, label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    print("image shape:",image_shape)
    print("label shape:",label_shape)
    image = tf.reshape(image, (image_shape[0], image_shape[1], 3))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 30], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 30], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 3))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_20(image, label):
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 20], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 20], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 1))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_30(image, label):
    image_shape = tf.shape(image)
    label_shape = tf.shape(label)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 30], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 30], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (label_shape[0], label_shape[1], 1))
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label

def rotate_dataset_10_tongue_and_vein(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 10], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 10], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 10], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label

def rotate_dataset_20_tongue_and_vein(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 20], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 20], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 20], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label

def rotate_dataset_30_tongue_and_vein(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1]))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 30], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 30], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 30], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label

def rotate_dataset_10_tongue_and_vein_RGB(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxximage shape:", image.get_shape().as_list())
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1],3))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 10], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 10], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 10], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 3))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label

def rotate_dataset_20_tongue_and_vein_RGB(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1],3))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 20], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 20], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 20], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 3))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label

def rotate_dataset_30_tongue_and_vein_RGB(image, tongue_label, vein_label):
    """
    for grayimage
    :param image:
    :param label:
    :return:
    """
    image_shape = tf.shape(image)
    tongue_label_shape = tf.shape(tongue_label)
    vein_label_shape = tf.shape(vein_label)
    image = tf.reshape(image, (image_shape[0], image_shape[1],3))
    tongue_label = tf.reshape(tongue_label, (tongue_label_shape[0], tongue_label_shape[1]))
    vein_label = tf.reshape(vein_label, (vein_label_shape[0], vein_label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 30], tf.uint8), tf.float32)
    rotated_tongue_label = tf.cast(tf.py_func(rotate_image_func, [tongue_label, 30], tf.uint8), tf.float32)
    rotated_vein_label = tf.cast(tf.py_func(rotate_image_func, [vein_label, 30], tf.uint8), tf.float32)
    rotated_image = tf.reshape(rotated_image, (tongue_label_shape[0], tongue_label_shape[1], 3))
    rotated_tongue_label = tf.reshape(rotated_tongue_label, (tongue_label_shape[0], tongue_label_shape[1], 1))
    rotated_vein_label = tf.reshape(rotated_vein_label, (vein_label_shape[0], vein_label_shape[1], 1))
    return rotated_image, rotated_tongue_label, rotated_vein_label



def rotate_dataset_315(image, label):
    label_shape = tf.shape(label)
    label = tf.reshape(label, (label_shape[0], label_shape[1]))
    rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 315], tf.uint8), tf.float32)
    rotated_label = tf.cast(tf.py_func(rotate_image_func, [label, 315], tf.uint8), tf.float32)
    rotated_label = tf.reshape(rotated_label, (label_shape[0], label_shape[1], 1))
    return rotated_image, rotated_label
# </editor-fold>


# <editor-fold desc="contract adjustment">
def adjust_contract_dataset_zero5(image, label):
    adjusted_image = tf.image.adjust_contrast(images=image, contrast_factor=0.5)
    return adjusted_image, label


def adjust_contract_dataset_zero6(image, label):
    adjusted_image = tf.image.adjust_contrast(images=image, contrast_factor=0.6)
    return adjusted_image, label


def adjust_contract_dataset_zero7(image, label):
    adjusted_image = tf.image.adjust_contrast(images=image, contrast_factor=0.7)
    return adjusted_image, label


def adjust_contract_dataset_zero8(image, label):
    adjusted_image = tf.image.adjust_contrast(images=image, contrast_factor=0.8)
    return adjusted_image, label


def adjust_contract_dataset_zero9(image, label):
    adjusted_image = tf.image.adjust_contrast(images=image, contrast_factor=0.9)
    return adjusted_image, label


def adjust_contract_random(image, label):
    adjusted_image = tf.image.random_contrast(image=image, lower=0.2, upper=0.9)
    return adjusted_image, label
# </editor-fold>


# data enhancement
# standardization
def standardization_norm(image, label):
    image = image/255.
    label = label/255.
    standardized_image = tf.image.per_image_standardization(image)
    return standardized_image, label

def standardization_norm_tongue_and_vein(image, tongue_label, vein_label):
    image = image/255.
    tongue_label = tongue_label/255.
    vein_label = vein_label / 255.
    standardized_image = tf.image.per_image_standardization(image)
    return standardized_image, tongue_label, vein_label

def only_norm(image, label):
    image = image/255.
    label = label/255.

    return image, label

def only_norm_tongue_and_vein(image, tongue_label, vein_label):
    image = image/255.
    tongue_label = tongue_label/255.
    vein_label = vein_label / 255.
    return image, tongue_label, vein_label

def only_norm_only_input(image):
    image = image/255.

    return image

def from_zero_one_to_minus_positive_one(image, labels):
    """
    :param image:  each input image
    :return:  output image with value from [0, 1] to [-1, 1]
    """
    with tf.name_scope("from_zero_one_to_minus_positive_one"):
        return image * 2 - 1, labels * 2 - 1

def from_zero_one_to_minus_positive_one_tongue_and_vein(image, tongue_label, vein_label):
    """
    :param image:  each input image
    :return:  output image with value from [0, 1] to [-1, 1]
    """
    with tf.name_scope("from_zero_one_to_minus_positive_one"):
        return image * 2 - 1, tongue_label * 2 - 1, vein_label * 2 - 1

def from_zero_one_to_minus_positive_one_only_input(image):
    """
    :param image:  each input image
    :return:  output image with value from [0, 1] to [-1, 1]
    """
    with tf.name_scope("from_zero_one_to_minus_positive_one_only_input"):
        return image * 2 - 1

def horizontal_flip_only_input(image):
    """
    :param image:  each input image
    :return:  output image with value from [0, 1] to [-1, 1]
    """
    with tf.name_scope("from_zero_one_to_minus_positive_one_only_input"):
        return tf.image.flip_left_right(image=image)

def minus_positive_one_to_zero_one(image, labels):
    """
    :param image:  each input image
    :return:  output image with value from [-1, 1] to [0, 1]
    """
    with tf.name_scope("from_zero_one_to_minus_positive_one"):
        return (image + 1) / 2, (labels + 1) / 2

# crop image according bounding box
# def crop_with_boundingbox_py(image, label):
#     """
#     This is a python function
#     :param image:  one input image with shape [height, width]
#     :param label:  mask/label of input image with shape [height, width]
#     :return:  return a cropped image which is according to the label of the input image
#     """
#     x, y, w, h = get_boundingBoxFromMask(label)
#     crop_image = image[y: y + h, x: x + w]
#     crop_label = label[y: y + h, x: x + w]
#     # resized_cropped_image = cv2.resize(crop_image, (label.shape[0], label.shape[1]), interpolation=cv2.INTER_LINEAR)
#     # resized_cropped_label = cv2.resize(crop_label, (label.shape[0], label.shape[1]), interpolation=cv2.INTER_LINEAR)
#     # return resized_cropped_image, resized_cropped_label
#     return crop_image, crop_label
#     # return resized_cropped_label

# def crop_with_boundingbox(image, label):
#     """
#     This is a tensorflow op using tf.py_func to translate python function
#     :param image:  one input image
#     :param label:  mask/label of input image
#     :return:  return a cropped image which is according to the label of the input image
#     """
#
#     label_shape = tf.shape(label)
#     label = tf.reshape(label, (label_shape[0], label_shape[1]))
#     # image = tf.reshape(image, (label_shape[0], label_shape[1]))
#     # resized_cropped_image, _ = tf.cast(tf.py_func(crop_with_boundingbox_py, [image, label], tf.uint8), tf.float32)
#     crop_image, crop_label= tf.cast(tf.py_func(crop_with_boundingbox_py, [image, label], tf.uint8), tf.float32)
#     # rotated_image = tf.cast(tf.py_func(rotate_image_func, [image, 30], tf.uint8), tf.float32)
#     label_shape = tf.shape(crop_label)
#     crop_image = tf.reshape(crop_image, (label_shape[0], label_shape[1], 1))
#     crop_label = tf.reshape(crop_image, (label_shape[0], label_shape[1], 1))
#     return crop_image, crop_label

# def crop_resize_boundingbox_in_image(image, label):
#     label_shape = tf.shape(label)
#     label = tf.reshape(label, (label_shape[0], label_shape[1]))
#     image = tf.reshape(image, (label_shape[0], label_shape[1]))
#     boxs = get_boundingBoxFromMask_tf(label)
#     label = tf.reshape(label, (1, label_shape[0], label_shape[1], 1))
#     image = tf.reshape(image, (1, label_shape[0], label_shape[1], 1))
#     boxs = tf.reshape(boxs, (1, 4))
#     print(boxs)
#     croped_image = tf.image.crop_and_resize(image=image, boxes=boxs, box_ind=[ 0 ], crop_size=[512, 640])
#     croped_label = tf.image.crop_and_resize(image=label, boxes=boxs, box_ind=[ 0 ], crop_size=[512, 640])
#     cropped_label = tf.reshape(croped_label, (label_shape[0], label_shape[1], 1))
#     cropped_image = tf.reshape(croped_image, (label_shape[0], label_shape[1], 1))
#     return cropped_image, cropped_label

# get bounding box from mask
def get_boundingBoxFromMask_tf(mask):
    """this is tensorflow op used for getting boundingbox from mask
    :param mask binary mask: shape with [image_height, image_width]
    :param num: number of generated box, if num=1 only contains orignal bounding box
    :return boxs : which contains [ymin_normalized, xmin_normalized, ymax_normalized, xmax_normalized]
    """
    zero = tf.constant(0, dtype=tf.float32)
    where = tf.not_equal(mask, zero)
    indices = tf.where(where)
    min_index = tf.argmin(indices, axis=0)
    max_index = tf.argmax(indices, axis=0)
    ymin_normalized = tf.cast(indices[min_index[0], 0] / 512, tf.float32)
    xmin_normalized = tf.cast(indices[min_index[1], 1] / 640, tf.float32)
    ymax_normalized = tf.cast(indices[max_index[0], 0] / 512, tf.float32)
    xmax_normalized = tf.cast(indices[max_index[1], 1] / 640, tf.float32)

    # xmin_normalized = np.random.uniform(low=0.001, high=xmax_normalized-0.001, size=(1)).astype(np.float32)
    # ymin_normalized = np.random.uniform(low=0.001, high=ymax_normalized - 0.001, size=(1)).astype(np.float32)

    xmin_normalized = tf.random_uniform(shape=[], minval=0.001, maxval=xmax_normalized-0.001)
    ymin_normalized = tf.random_uniform(shape=[], minval=0.001, maxval=ymax_normalized-0.001)
    w = tf.random_uniform(shape=[], minval=0.2, maxval=1-xmin_normalized-0.001)
    # h = tf.random_uniform(shape=[], minval=0.001, maxval=1-ymin_normalized-0.001)
    ratio =  tf.random_uniform(shape=[], minval=0.75, maxval=1.33)
    ymax_normalized = ymin_normalized + w/ratio
    xmax_normalized = xmin_normalized+w
    boxs = [[ymin_normalized, xmin_normalized, ymax_normalized, xmax_normalized]]
    # boxs = [ymin_normalized_new, xmin_normalized_new, ymax_normalized_new, xmax_normalized_new]

    return boxs


def get_boundingBoxFromMask(mask):
    """
    :param mask: the mask used to define the bounding box
    :return: the bounding box parameters(x, y, w, h)
    """
    x, y, w, h = cv2.boundingRect(mask)
    return x, y, w, h

