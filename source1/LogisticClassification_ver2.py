import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('04train.txt', dtype='float32')
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

print(x_data)
print(y_data) 

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))

h = tf.matmul(X, W)
hypothesis = tf.div(1., 1. + tf.exp(-h))
# hypothesis = tf.sigmoid(h)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))


a = tf.Variable(0.1)  
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(5001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

print('-----------------------------------------')
print(sess.run(hypothesis,feed_dict={X:[[1,2,2]]}))







print(sess.run(hypothesis, feed_dict={X: [[1, 2, 2]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1, 5, 5]]}) > 0.5)
print(sess.run(hypothesis, feed_dict={X: [[1,4,2],[1, 0,10]]}) > 0.5)






predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y)
                                  ,dtype=tf.float32))
print(sess.run(accuracy,feed_dict={X:x_data, Y:y_data} ))





