import os
import math
import numpy as np
import tensorflow as tf
import time
from PIL import Image

VGG_MEAN = [103.939,116.779,123.68]
class VGGNet:
    """Build VGG_16 net structure"""
    def __init__(self,data_dict):
        self.data_dict = data_dict#参数作为全局变量

#提取卷积参数
    def get_conv_filter(self,name):
        return tf.constant(self.data_dict[name][0],name='conv')

    def get_fc_weight(self,name):
        return tf.constant(self.data_dict[name][0],name='fc')

    def get_bias(self,name):
        return tf.constant(self.data_dict[name][1],name='bias')
#创建卷积层
    def conv_layer(self,x,name):
        # 命名空间
        with tf.name_scope(name):
          conv_w = self.get_conv_filter(name)
          conv_b = self.get_bias(name)
          h = tf.nn.conv2d(x,conv_w,[1,1,1,1],padding='SAME')
          # 使用bias_add 将bias添加到value中
          h = tf.nn.bias_add(h,conv_b)
          h = tf.nn.relu(h)
          return h

    def pooling_layer(self,x,name):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                              padding="SAME",name=name)

    def fc_layer(self,x,name,activation=tf.nn.relu):
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)

            h = tf.matmul(x,fc_w)
            h = tf.nn.bias_add(h,fc_b)
            if activation is None:
                return h
            else:
                return activation(h)

    def flatten_layer(self,x,name):
        with tf.name_scope(name):
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim*=d
            x = tf.reshape(x,[-1,dim])
            return x


    def build(self,x_rgb):
        start_time = time.time()
        print("building model")

        # 处理输入图像减去均值
        r,g,b = tf.split(x_rgb,[1,1,1],axis=3)
        x_bgr = tf.concat([b-VGG_MEAN[0],
                          g - VGG_MEAN[1],
                          r - VGG_MEAN[2]],
                          axis=3)
        assert x_bgr.get_shape().as_list()[1:] == [224,224,3]

        self.conv1_1 = self.conv_layer(x_bgr,'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1,'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2,'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')
        print('buliding model finished:%4ds'%(time.time()-start_time))

'''
        self.flatten5 = self.flatten_layer(self.pool5,'flatten')
        self.fc6 = self.fc_layer(self.flatten5,'fc6')
        self.fc7 = self.fc_layer(self.fc6,'fc7')
        self.fc8 = self.fc_layer(self.fc7,'fc8',activation=None)

        self.prob = tf.nn.softmax(self.fc8,name='prob')
'''




vgg16_npy_pyth = './data/vgg16.npy'
content_img_path = './source_images/gugong.jpg'
style_img_path = './source_images/xingkong.jpeg'

num_steps = 100
learning_rate = 10

lambda_c = 0.1
lambda_s = 500

output_dir = './run_style_transfer'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# data_dict = np.load(vgg16_npy_pyth,encoding='latin1').item()
# vgg16_for_result = VGGNet(data_dict)
# content = tf.placeholder(tf.float32,shape=[1,224,224,3])
# vgg16_for_result.build(content)


def initial_result(shape,mean,stddev):
    initial = tf.truncated_normal(shape,mean=mean,stddev=stddev)
    return tf.Variable(initial)

def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img) #(224,224,3)
    np_image = np.asarray([np_img],dtype=np.int32) #(1,224,224,3)
    return np_image


def gram_matrix(x):
    '''计算gram矩阵 
    shape:[1,width,height,channel]
    x 图片
    '''
    b,w,h,ch = x.get_shape().as_list()
    features = tf.reshape(x,[b,h*w,ch])
    # 将[h*w,ch] 看做二维举证 选择两列 计算相似度 ->[ch ,h *w]*[h*w,ch]  -->[ch,ch]
    gram = tf.matmul(features,features,adjoint_a=True)/tf.constant(ch*w*h,tf.float32)
    return gram



result = initial_result((1,224,224,3),127.5,20)
content_val = read_img(content_img_path)
style_val = read_img(style_img_path)

content = tf.placeholder(tf.float32,shape=[1,224,224 ,3])
style = tf.placeholder(tf.float32,shape=[1,224,224,3])

data_dict = np.load(vgg16_npy_pyth,encoding='latin1').item()
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

vgg_for_content.build(content)
vgg_for_style.build(style)
vgg_for_result.build(result)

content_features = [vgg_for_content.conv1_2,
                    vgg_for_content.conv2_2,
                    # vgg_for_content.conv3_3,
                    # vgg_for_content.conv4_3,
                    # vgg_for_content.conv5_3
                    ]
result_content_features = [
                    vgg_for_result.conv1_2,
                    vgg_for_result.conv2_2,
                    # vgg_for_result.conv3_3,
                    # vgg_for_result.conv4_3,
                    # vgg_for_result.conv5_3
                    ]
#feature_size,[1,width,height,channel]
style_features = [
                    # vgg_for_style.conv1_2,
                    # vgg_for_style.conv2_2,
                    # vgg_for_style.conv3_3,
                    vgg_for_style.conv4_3,
                    # vgg_for_style.conv5_3
                    ]

style_gram = [gram_matrix(feature) for feature in style_features]


result_style_features = [
                    # vgg_for_result.conv1_2,
                    # vgg_for_result.conv2_2,
                    # vgg_for_result.conv3_3,
                    vgg_for_result.conv4_3,
                    # vgg_for_result.conv5_3
                    ]
result_style_gram = [gram_matrix(feature) for feature in result_style_features]


content_loss = tf.zeros(1,tf.float32)
# 特征维度shape[1,width,height,channel]
for c,c_ in zip(content_features,result_content_features):
    content_loss += tf.reduce_mean((c - c_)**2,[1,2,3])  # 在123通道做损失

style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])



loss = content_loss*lambda_c + style_loss*lambda_s
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_value,content_loss_value,style_loss_value,_ = sess.run([loss,content_loss,style_loss,train_op],feed_dict={
            content:content_val,
            style:style_val
        }
                                                                    )

        print('step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f'
              % (step + 1,
                 loss_value[0],
                 content_loss_value[0],
                 style_loss_value[0]))

        result_img_path = os.path.join(output_dir,"result-%05d.jpg"%(step+1))

        result_val = result.eval(sess)[0]
        result_val = np.clip(result_val,0,255)

        img_arr = np.array(result_val,np.uint8)
        img = Image.fromarray(img_arr)
        img.save(result_img_path)











