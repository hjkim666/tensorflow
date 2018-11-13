import tensorflow as tf 
import numpy as np 

xy = np.loadtxt('iris_training.csv', delimiter=',', dtype='float32')
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32, [None,4])
Y = tf.placeholder(tf.int32,[None,1])

Y_one_hot = tf.one_hot(Y, 3)
Y_one_hot = tf.reshape(Y_one_hot, [-1,3])
print(Y_one_hot)

W = tf.Variable(tf.random_uniform([4,3], -1, 1))
b = tf.Variable(tf.random_uniform([3], -1, 1))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    
    a = sess.run(hypothesis, feed_dict={X: [[6.4,2.8,5.6,2.2]]})
    print("a :", a, sess.run(tf.argmax(a, 1)))
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))  
