import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

studentID_name = "studtentID_name"
dropout_rate = 0.75  # 드롭아웃 비율 설정
train_epoch = 800

def build_CNN_classifier(x):

    W_conv0 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=5e-2))
    b_conv0 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv0 = tf.nn.leaky_relu(tf.nn.conv2d(x, W_conv0, strides=[1, 1, 1, 1], padding='SAME') + b_conv0)

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.leaky_relu(tf.nn.conv2d(h_conv0, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv1_drop = tf.nn.dropout(h_pool1, keep_prob)

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1_drop, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_drop = tf.nn.dropout(h_pool2, keep_prob)

    W_conv1_5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv1_5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv1_5 = tf.nn.leaky_relu \
        (tf.nn.conv2d(h_conv2_drop, W_conv1_5, strides=[1, 1, 1, 1], padding='SAME') + b_conv1_5)

    W_conv3 = tf.Variable(tf.truncated_normal(shape=[4, 4, 128, 256], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]))
    h_conv3= tf.nn.leaky_relu(tf.nn.conv2d(h_conv1_5, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_drop = tf.nn.dropout(h_pool3, keep_prob)

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 256, 512], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
    h_conv_flat = tf.reshape(h_conv3_drop, [-1, 4* 4 * 256])
    h_fc1 = tf.nn.leaky_relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

    W_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 128], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_fc2 = tf.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, 0.5)

    W_fc4 = tf.Variable(tf.truncated_normal(shape=[128, 2], stddev=5e-2))
    b_fc4 = tf.Variable(tf.constant(0.1, shape=[2]))
    logits = tf.matmul(h_fc2_drop, W_fc4) + b_fc4

    return logits


def load_image(dir):
    folders = glob.glob(dir + "*")

    label = []
    train_x_data = []
    train_y_label = []

    for i in range(len(folders)):
        label.append(folders[i].split("\\")[1])
        image_dir = glob.glob(dir + label[i] + "/*.jpg")

        for j in range(len(image_dir)):
            train_x_data.append(plt.imread(image_dir[j]))

            if i == 0:
                train_y_label.append((1, 0))
            else:
                train_y_label.append((0, 1))

    train_x_data = np.array(train_x_data, dtype=np.int32).reshape(-1, 32, 32, 3)
    train_y_label = np.array(train_y_label, dtype=np.int32).reshape(-1, 2)
    label = np.array(label)

    idx = np.arange(train_x_data.shape[0])
    np.random.shuffle(idx)
    train_x_data = train_x_data[idx]
    train_y_label = train_y_label[idx]

    return train_x_data, train_y_label, label


dir = "./dataset/"
train_x_data, train_y_label, label = load_image(dir)

validation_x_data = train_x_data[1000:1600]
validation_y_label = train_y_label[1000:1600]
train_x_data = train_x_data[:1000]
train_y_label = train_y_label[:1000]

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

logits = build_CNN_classifier(x)
y_pred = tf.nn.softmax(logits, name="y_pred")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
tf.summary.scalar('loss', loss)
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_dir = studentID_name
saver = tf.train.Saver()
checkpoint_path = os.path.join(save_dir, "model")
ckpt = tf.train.get_checkpoint_state(save_dir)

sess = tf.Session()
# 모든 변수들을 초기화한다.
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log', sess.graph)

# 10000 Step만큼 최적화를 수행합니다.
for i in range(train_epoch):

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 99:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: train_x_data, y: train_y_label, keep_prob: 1.0})
        loss_print = loss.eval(session=sess, feed_dict={x: train_x_data, y: train_y_label, keep_prob: 1.0})

        print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
        saver.save(sess, checkpoint_path, global_step=i)
    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    sess.run(train_step, feed_dict={x: train_x_data, y: train_y_label, keep_prob: dropout_rate})
    summary = sess.run(merged, feed_dict={x: train_x_data, y: train_y_label, keep_prob: dropout_rate})
    tensorboard_writer.add_summary(summary, i)

test_accuracy = accuracy.eval(session=sess, feed_dict={x: validation_x_data, y: validation_y_label, keep_prob: 1.0})
loss_print = loss.eval(session=sess, feed_dict={x: validation_x_data, y: validation_y_label, keep_prob: 1.0})
print("검증 데이터 정확도: %f, 손실 함수(loss): %f" % (test_accuracy, loss_print))

sess.close()