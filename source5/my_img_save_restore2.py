import tensorflow as tf
import random
import numpy as np
import os
tf.set_random_seed(777) 

cwd = os.getcwd()
loadpath = cwd + "/data/myData.npz"
l = np.load(loadpath)
print (l.files)

imgsize = [28,28]
use_gray =1 
data_name = "myData"

# PARSE LOADED DATA #####################
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
use_gray = l['use_gray']
ntrain = trainimg.shape[0] #row
nclass = trainlabel.shape[1] #col
ntest = testimg.shape[0]

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

# L1 Conv shape=(?, 28, 28, 32)
#    Pool     ->(?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob)

# L2 Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L2 = tf.nn.dropout(L2, keep_prob)

#  (?, 7, 7, 64) Reshape  ->(?, 256)
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L4 = tf.matmul(L3, W4)
L4 = tf.nn.relu(L4)

W5 = tf.Variable(tf.random_normal([256, 4], stddev=0.01))
hypothesis = tf.matmul(L4, W5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(ntrain/ batch_size)

for epoch in range(1000):
    total_cost = 0

    for i in range(total_batch):
        randidx = np.random.randint(ntrain, size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
 

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: testimg.reshape(-1, 28, 28, 1), Y: testlabel, keep_prob: 1}))

# Get one and predict
r = random.randint(0, ntest - 1)
print("Label: ", sess.run(tf.argmax(testlabel[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: testimg[r:r + 1].reshape(-1, 28, 28, 1), keep_prob: 1}))

