# import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# v1 = tf.constant(1,name="value1")
# v2 = tf.constant(2,name="value2")
#
# add_op = tf.add(v1,v2,name="add_op_name")
#
# with tf.Session() as sess:
#     result = sess.run(add_op)
#     print("1+2=%0.f"%result)
#
# tf.get_default_graph
# print(v1)
# print(v2)
# import tensorflow as tf
#
# with tf.name_scope("a_name_scope") as myscope:
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(var1.name)        # var1:0
#     print(sess.run(var1))   # [ 1.]

image = np.array([[[1,2,3], [4,5,6]], [[1,1,1], [1,1,1]],[[1,1,1], [1,1,1]]])

