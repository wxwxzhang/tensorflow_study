import tensorflow as tf
import os
import pickle
import numpy as np

"""    保存模型(第三方)
       更新模型
       保持层不变
"""

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_labels.append(labels)
            all_data.append(data)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0

        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("没有案例了")
        if end_indicator > self._num_examples:
            raise Exception("遍历结束")
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1, 3, 32, 32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

conv1_1 = tf.layers.conv2d(x_image,
                           32,
                           (3, 3),
                           padding="same",
                           activation=tf.nn.relu,
                           trainable=False,
                           name='conv1_1')

conv1_2 = tf.layers.conv2d(conv1_1,
                           32,
                           (3, 3),
                           padding="same",
                           activation=tf.nn.relu,
                           trainable=False,
                           name="conv1_2")
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2, 2),
                                   (2, 2),
                                   name='pool1')
conv2_1 = tf.layers.conv2d(pooling1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           trainable=False,
                           name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           trainable=False,
                           name='conv2_2')
# 8 * 8
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool2')

conv3_1 = tf.layers.conv2d(pooling2,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           trainable=False,
                           name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,
                           32,  # output channel number
                           (3, 3),  # kernel size
                           padding='same',
                           activation=tf.nn.relu,
                           trainable=False,
                           name='conv3_2')
# 4 * 4 * 32
pooling3 = tf.layers.max_pooling2d(conv3_2,
                                   (2, 2),  # kernel size
                                   (2, 2),  # stride
                                   name='pool3')
# [None, 4 * 4 * 32]
flatten = tf.contrib.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

predict = tf.argmax(y_, 1)
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('trian_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
'''给一个变量的统计量summary'''


def variable_summary(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('mean', tf.reduce_max(var))
        # 直方图
        tf.summary.histogram('histogram', var)


with tf.name_scope('summary'):
    variable_summary(conv1_1, 'conv1-1')
    variable_summary(conv1_2, 'conv1-2')
    variable_summary(conv2_1, 'conv2-1')
    variable_summary(conv2_2, 'conv2-2')
    variable_summary(conv3_1, 'conv3-1')
    variable_summary(conv3_2, 'conv3-2')

'''loss :<10,1.1> <20,1.8>'''
loss_summary = tf.summary.scalar('loss', loss)
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
'''归一化的逆过程  真实的图像'''
source_image = (x_image + 1) * 127.5

inputs_summary = tf.summary.image('input_image', source_image)

merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

LOG_DIR = '.'
run_label = './run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)

if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')

if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

model_dir = os.path.join(run_dir,'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

saver = tf.train.Saver()
model_name='ckp-01000'
model_path = os.path.join(model_dir,model_name)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 1000
test_steps = 100

output_summary_every_steps = 100
output_model_every_steps = 100

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data, fixed_test_batch_labels = test_data.next_batch(batch_size)

    if os.path.exists(model_path + '.index'):
        saver.restore(sess, model_path)
        print('model restored from %s' % model_path)
    else:
        print('model %s does not exist' % model_path)


    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i + 1) % output_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(merged_summary)

        eval_ops_results = sess.run(
            eval_ops,
            feed_dict={
                x: batch_data,
                y: batch_labels})
        loss_val, acc_val = eval_ops_results[0:2]
        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str, i + 1)
            test_summary_str = sess.run([merged_summary_test],
                                        feed_dict={
                                            x: fixed_test_batch_data,
                                            y: fixed_test_batch_labels,
                                        })[0]
            test_writer.add_summary(test_summary_str, i + 1)

        if (i + 1) % 100 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f'
                  % (i + 1, loss_val, acc_val))
        if (i + 1) % 1000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step:%d, acc: %4.5f' % (i + 1, test_acc))

        if(i+1)%output_model_every_steps==0:
            saver.save(sess, os.path.join(model_dir, 'ckp-%05d' % (i+1)))
            print('model saved to ckp-%05d' % (i+1))