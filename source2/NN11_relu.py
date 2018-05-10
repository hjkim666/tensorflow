import tensorflow as tf
import numpy as np

xy = np.loadtxt('07train.txt')
x_data =  xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.random_uniform([2,  10], -1.0, 1.0), name='weight1')
# W1 = tf.get_variable("W1",shape=[2, 10]
#                  ,initializer=tf.contrib.layers.xavier_initializer(2,10))

w2 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight2')
w3 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight3')
w4 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight4')
w5 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight5')
w6 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight6')
w7 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight7')
w8 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight8')
w9 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight9')
w10 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name='weight10')
w11 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name='weight11')

b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b3 = tf.Variable(tf.zeros([10]), name="Bias3")
b2 = tf.Variable(tf.zeros([10]), name="Bias2")
b4 = tf.Variable(tf.zeros([10]), name="Bias4")
b5 = tf.Variable(tf.zeros([10]), name="Bias5")
b6 = tf.Variable(tf.zeros([10]), name="Bias6")
b7 = tf.Variable(tf.zeros([10]), name="Bias7")
b8 = tf.Variable(tf.zeros([10]), name="Bias8")
b9 = tf.Variable(tf.zeros([10]), name="Bias9")
b10 = tf.Variable(tf.zeros([10]), name="Bias10")
b11 = tf.Variable(tf.zeros([1]), name="Bias11")

# L1 = tf.sigmoid(tf.matmul(X, w1) + b1)
# L2 = tf.sigmoid(tf.matmul(L1, w2) + b2)
# L3 = tf.sigmoid(tf.matmul(L2, w3) + b3)
# L4 = tf.sigmoid(tf.matmul(L3, w4) + b4)
# L5 = tf.sigmoid(tf.matmul(L4, w5) + b5)
# L6 = tf.sigmoid(tf.matmul(L5, w6) + b6)
# L7 = tf.sigmoid(tf.matmul(L6, w7) + b7)
# L8 = tf.sigmoid(tf.matmul(L7, w8) + b8)
# L9 = tf.sigmoid(tf.matmul(L8, w9) + b9)
# L10 = tf.sigmoid(tf.matmul(L9, w10) + b10)
L1 = tf.nn.relu(tf.matmul(X, w1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, w2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, w3) + b3)
L4 = tf.nn.relu(tf.matmul(L3, w4) + b4)
L5 = tf.nn.relu(tf.matmul(L4, w5) + b5)
L6 = tf.nn.relu(tf.matmul(L5, w6) + b6)
L7 = tf.nn.relu(tf.matmul(L6, w7) + b7)
L8 = tf.nn.relu(tf.matmul(L7, w8) + b8)
L9 = tf.nn.relu(tf.matmul(L8, w9) + b9)
L10 = tf.nn.relu(tf.matmul(L9, w10) + b10)
hypothesis = tf.sigmoid(tf.matmul(L10, w11) + b11)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1 - hypothesis))

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w1), sess.run(w2))

    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction], feed_dict={X: x_data, Y: y_data}))
    print("accuracy", accuracy.eval({X: x_data, Y: y_data}))
