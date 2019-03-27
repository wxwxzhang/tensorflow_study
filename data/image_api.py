#resize
#crop
#flip
#brightness contrast

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# name = './gugong.jpg'

#resize
#tf.image.resize_area
#tf.image.resize_bicubic
#tf.image.resize_nearest_neighbor
name = './gugong.jpg'
img_string = tf.read_file(name)
img_decoded = tf.image.decode_image(img_string)
img_decoded = tf.reshape(img_decoded,[1,365,600,3])

resize_img = tf.image.resize_bicubic(img_decoded,[730,1200])
sess = tf.Session()
img_decode_val = sess.run(resize_img)
img_decode_val = img_decode_val.reshape((730,1200,3))
img_decode_val = np.asarray(img_decode_val,np.uint8)
print(img_decode_val.shape)
plt.imshow(img_decode_val)
plt.show()