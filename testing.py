# ckpt로 저장된 모델 불러오기

import glob
import matplotlib.pyplot as plt
import numpy as np

import glob as glob
import tensorflow as tf

image_dir = "test1.jpg"
dir = "./ckpt/"

test_x_data = plt.imread(image_dir)
test_y_label = [(1 ,0)]

test_x_data = np.array(test_x_data, dtype=np.int32).reshape(-1 ,32 ,32 ,3)
test_y_label = np.array(test_y_label, dtype=np.int32).reshape(-1 ,2)

models_dir = glob.glob(dir +"*")

test_score = []

for i in range(len(models_dir)):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(models_dir[i])

        new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +".meta")
        new_saver.restore(sess, ckpt.model_checkpoint_path)

        tf.get_default_graph()

        x = sess.graph.get_tensor_by_name("x:0")
        y = sess.graph.get_tensor_by_name("y:0")
        keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
        y_pred = sess.graph.get_tensor_by_name("y_pred:0")

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_accuracy = accuracy.eval(session=sess, feed_dict={x: test_x_data, y: test_y_label, keep_prob: 1.0})
        test_score.append((models_dir[i].split('\\')[1] ,train_accuracy))