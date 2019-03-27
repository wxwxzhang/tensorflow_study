import tensorflow as tf
import os
def weight_init(shape,name):
    return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1))

def bias_init(shape,name):
    return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.0))

def con2d(x,conv_w):
    return tf.nn.conv2d(x,conv_w,strides=[1,1,1,1],padding='VALID')

def max_pool(x,size):
    return tf.nn.max_pool(x, ksize=[1,size,size,1], strides = [1,size,size,1], padding='VALID')

