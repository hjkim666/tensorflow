#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility

idx2char = ['아', '리', '랑']
# 아리아리아리 -> 리아리아리랑
x_data = [[0, 1, 0, 1, 0, 1]]   # 아리아리아리랑
x_one_hot = [[[1, 0, 0],   # 아 0
              [0, 1, 0],   # 리 1
              [1, 0, 0],   # 아 0
              [0, 1, 0],   # 리 1
              [1, 0, 0],   # 아 0
              [0, 1, 0]]]  # 리 1
y_data = [[1, 0, 1, 0, 1, 2]]    # 리아리아리랑

input_dim = 3  # one-hot size
hidden_size = 3  # output from the LSTM.
batch_size = 1   # one sentence
sequence_length = 6  # |리아리아리랑| == 6

X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))




