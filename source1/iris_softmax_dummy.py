import tensorflow as tf 
import numpy as np 
import pandas as pd 

xy = np.loadtxt('iris_training.csv', delimiter=',', dtype='float32')
x_data = xy[:,0:-1]
y_data = xy[:,-1]
y_data = pd.get_dummies(y_data).values

xy = np.loadtxt('iris_test.csv', delimiter=',', dtype='float32')
testX = xy[:,0:-1]
testY = xy[:,-1]
testY = pd.get_dummies(testY).values

X = tf.placeholder(tf.float32, [None,4])
Y = tf.placeholder(tf.float32,[None,3])

W = tf.Variable(tf.random_uniform([4,3], -1, 1))
b = tf.Variable(tf.random_uniform([3], -1, 1))

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    
    a = sess.run(hypothesis, feed_dict={X: [[6.4,2.8,5.6,2.2]]})
    print("a :", a, sess.run(tf.argmax(a, 1)))
    
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("training accuracy",sess.run(accuracy, feed_dict={X:x_data, Y:y_data}))  
    print("test accuarcy",sess.run(accuracy, feed_dict={X:testX, Y:testY}))  