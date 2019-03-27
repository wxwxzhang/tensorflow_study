import numpy as np
import struct
import random

class MnistData:
    def __init__(self,train_image_path,train_label_path,test_image_path,test_label_path):
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path

    def get_data(self,data_type):
        if data_type == 0:
            image_path = self.train_image_path
            label_path = self.train_label_path
        else:
            image_path = self.test_image_path
            label_path = self.test_label_path

        with open(image_path,'rb') as file1:
            image_file = file1.read()

        with open(image_path,'rb') as file2:
            label_file = file2.read()

        label_index = 0
        image_index = 0

        labels = []
        images = []

        # 读取训练集图像数据文件的文件信息
        magic ,num_of_datasets,rows,columns = struct.unpack_from(">iiii",image_file,image_index)
        image_index += struct.calcsize(">iiii")
        # 读取一个图像的所有像素值
        for i in range(num_of_datasets):
            temp = struct.unpack_from('>784B',image_file,image_index)
            #将读取的像素值转为(28*28)的矩阵
            temp = np.reshape(temp,(28,28))

            #归一化处理
            temp = temp/225

            images.append(temp)
            image_index += struct.calcsize(">784B")

        # 跳过描述信息

        label_index += struct.calcsize('>ii')

        labels = struct.unpack_from('>'+str(num_of_datasets)+'B',label_file,label_index)

        #ont_hot
        labels = np.eye(10)[np.array(labels)]

        return images,labels

    def get_batch(self,batch_size):

        #刚开始训练或者训练一轮结束后 打乱数据集的顺序
        if self.got_batch == self.num_of_batch:
            train_list = list(zip(self.train_images,self.train_labels))
            random.shuffle(train_list)
            self.train_images,self.train_labels = zip(*train_list)
            # 重置辅助变量
            self.num_of_batch = 6000/batch_size
            self.got_batch = 0

        #获取一个batch size的训练数据
        train_images = self.train_images[self.got_batch*batch_size:(self.got_batch+1)*batch_size]

        train_labels = self.train_labels[self.got_batch*batch_size:(self.got_batch+1)*batch_size]

        self.get_batch += 1

        return train_images,train_labels




