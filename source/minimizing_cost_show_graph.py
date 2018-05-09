import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_vals = []
cost_vals = []

for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_vals.append(curr_W)
    cost_vals.append(curr_cost)

plt.plot(W_vals, cost_vals)
plt.show()
