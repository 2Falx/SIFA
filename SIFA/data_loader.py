###import tensorflow as tf
import tensorflow.compat.v1 as tf ###
tf.disable_v2_behavior() ###
import json

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])


def _decode_samples(image_list, shuffle=False):
    decomp_feature = {
        #Tensorflow 2.x ==> tf.io.FixedLenFeature
        #Tensorflow 1.x ==> tf.FixedLenFeature
        
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 3]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

    #raw_size = [352, 352, 3]
    #volume_size = [352, 352, 3]
    #label_size = [352, 352, 1] # the label has size [256,256,3] in the preprocessed data, but only the middle slice is used
    volume_size = [352, 352, 1]
    label_size = [352, 352, 2]

    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, volume_size)#, raw_size)
    #(352,352,2)
       
    #data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)#, volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, label_size)#, raw_size) 
    #(352,352,2)
    
    #label_vol = tf.slice(label_vol, [0, 0, 1], label_size)#, label_size)

    #batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
    batch_y = tf.cast(tf.squeeze(label_vol), tf.uint8)
    
    #return tf.expand_dims(data_vol[:, :, 1], axis=2), batch_y
    return data_vol, batch_y


def _load_samples(source_pth, target_pth):

    with open(source_pth, 'r') as fp:
        rows = fp.readlines()
    imagea_list = [row[:-1] for row in rows]

    with open(target_pth, 'r') as fp:
        rows = fp.readlines()
    imageb_list = [row[:-1] for row in rows]

    data_vola, label_vola = _decode_samples(imagea_list, shuffle=True)
    data_volb, label_volb = _decode_samples(imageb_list, shuffle=True)
    return data_vola, data_volb, label_vola, label_volb


def load_data(source_pth, target_pth, do_shuffle=True):

    image_i, image_j, gt_i, gt_j = _load_samples(source_pth, target_pth)

    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    """if 'mr' in source_pth:
        image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    elif 'ct' in source_pth:
        image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)

    if 'ct' in target_pth:
        image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    elif 'mr' in target_pth:
        image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)"""

    # Batch
    if do_shuffle is True:
        images_i, images_j, gt_i, gt_j = tf.train.shuffle_batch([image_i, image_j, gt_i, gt_j], BATCH_SIZE, 500, 100)
    else:
        images_i, images_j, gt_i, gt_j = tf.train.batch([image_i, image_j, gt_i, gt_j], \
                                                        batch_size=BATCH_SIZE, num_threads=1, capacity=500)

    return images_i, images_j, gt_i, gt_j