from num_reco.mnist_reco import MnistData
import tensorflow as tf

sess = tf.InteractiveSession()

#训练集,测试集文件路径
train_image_path = './MNIST_data/train-images-idx3-ubyte'
train_label_path = './MNIST_data/train-labels-idx1-ubyte'

test_image_path = './MNIST_data/t10k-images-idx3-ubyte'
test_label_path = './MNIST_data/t10k-labels-idx1-ubyte'


# 训练总轮数
epochs = 10
# 每个batch的大小
batch_size = 100
# 学习率
learning_rate = 0.2

x = tf.placeholder(tf.float32,[None,28,28])

#定义权重矩阵和偏置项
W = tf.Variable(tf.zeros([28*28,10]))
b = tf.Variable(tf.zeros([10]))

#样本的真实标签
y_ = tf.placeholder(tf.float32,[None,10])
#使用softmax函数将网络的输出转化为结果
y = tf.nn.softmax(tf.matmul(tf.reshape(x,[-1,28*28]),W)+b)

#损失函数和优化器

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),axis=1))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#比较预测结果和真实类标
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


data = MnistData(train_image_path,train_label_path,test_image_path,test_label_path)
#初始化模型参数
init = tf.global_variables_initializer().run()

#开始训练
for i in range(epochs):
    for j in range(100):
        batch_x ,batch_y = data.get_batch(batch_size)

        train_step.run({x:batch_x,y_:batch_y})


print(accuracy.eval({x:data.test_images,y_:data.test_labels}))

