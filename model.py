import numpy as np
import tensorflow as tf




def FSRCNN(input):
    d = 32
    s = 8
    m = 2
    
    with tf.variable_scope('feature_extraction'):
        conv = conv_layer(input, 5, 3, d, padding='VALID', name='layer_1')
        
    with tf.variable_scope('shrinking'):
        conv = conv_layer(conv, 1, d, s, padding='VALID', name='layer_2')
        
    with tf.variable_scope('non_linear_mapping'):
        for m in range(m):
            conv = conv_layer(conv, 3, s, s, padding='VALID', name='conv2d_{:}'.format(m))

    
    with tf.variable_scope('expanding'):
        conv = conv_layer(conv, 1, s, d, padding='VALID', name='layer_4')
    
    with tf.variable_scope('deconvolution'):
        conv = conv_transpose(conv, 9, 3, d, strides=[1,2,2,1], padding='SAME', name='layer_5')        
    summary = tf.summary.merge_all()
    
    return (conv, summary)

def SRCNN(input):
    upscale = 2

    conv = tf.image.resize_nearest_neighbor(input, tf.convert_to_tensor([64,64]))
    
    conv = conv_layer(conv, 9, 3, 64, padding='VALID', name='feature_extraction', activation=None)
    
    conv = conv_layer(conv, 1, 64, 32, padding='VALID', name='bottleneck', activation=tf.nn.relu)
    
    conv = conv_layer(conv, 5, 32, 3, padding='VALID', name='reconstruction', activation=None)

    summary = tf.summary.merge_all()

    return (conv, summary)
    
def conv_layer(input, k, in_channel, out_channel, padding='valid', activation=None, name=None):
    if name is not None:
        k_name = name+'/kernel'
        b_name = name+'/bias'
        
        w = tf.get_variable(k_name, [k, k, in_channel, out_channel], initializer=tf.initializers.glorot_uniform())
        b = tf.get_variable(b_name, initializer=tf.zeros([out_channel]))
        
        tf.summary.histogram(k_name, w)
        tf.summary.histogram(b_name, b)
        
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding=padding, name=name+'/conv2d_op')
        conv = tf.nn.bias_add(conv, b, name=name+'/bias_add_op')
        #conv = prelu(conv, name+'prelu_acts')
        if activation is not None:
            conv = activation(conv)
            tf.summary.histogram('prelu_acts', conv)

    return conv

def conv_transpose(input, k, in_channel, out_channel, strides, padding='valid', activation=None, name=None):
    if name is not None:
        k_name = name+'/kernel'
        b_name = name+'/bias'
        
        w = tf.get_variable(k_name, [k, k, in_channel, out_channel], initializer=tf.initializers.glorot_uniform())
        b = tf.get_variable(b_name, initializer=tf.zeros([in_channel]))
        
        tf.summary.histogram(k_name, w)
        tf.summary.histogram(b_name, b)
        
        output_shape = input.get_shape().as_list()
        output_shape[0] = 256
        output_shape[1], output_shape[2] = output_shape[1]*strides[1], output_shape[2]*strides[2]
        output_shape[3] = in_channel
        output_shape = tf.TensorShape(output_shape)
        
        conv = tf.nn.conv2d_transpose(input, w, output_shape, strides, padding=padding, name=name+'/conv2d_op')
        print(conv.get_shape().as_list())
        conv = tf.nn.bias_add(conv, b, name=name+'/bias_add_op')

        if activation is not None:
            conv = activation(conv)
            tf.summary.histogram('activations', conv)

    return conv

def prelu(_x, name):
    
    alphas = tf.get_variable('prelu_alpha_{:}'.format(name), _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                                dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = tf.multiply(alphas, (_x - abs(_x))) * 0.5
    tf.summary.histogram('prelu_alpha_{:}'.format(name), alphas)

    return pos + neg




    
