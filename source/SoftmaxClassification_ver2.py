import tensorflow as tf
import numpy as np

xy = np.loadtxt('05train.txt', dtype='float32')
x_data =xy[:,0:3]
y_data =xy[:,3:]

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.01
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(5001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data})
                  , sess.run(W))

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print("a :", a, sess.run(tf.argmax(a, 1)))

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print("b :", b, sess.run(tf.argmax(b, 1)))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print("c :", c, sess.run(tf.argmax(c, 1)))

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))    