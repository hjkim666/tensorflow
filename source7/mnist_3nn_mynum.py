from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
import matplotlib.pylab as plt
tf.set_random_seed(777)

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1",shape=[784, 256]
                 ,initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2",shape=[256, 256]
                     ,initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3",shape=[256, 10]
                     ,initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([10]))

dropout_rate = tf.placeholder(tf.float32)
_L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(_L1, keep_prob=dropout_rate)
_L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(_L2, keep_prob=dropout_rate)
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) 
train = tf.train.AdamOptimizer(0.001).minimize(cost)

tf.global_variables_initializer().run()

for i in range(5500):  #5500
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train.run({X: batch_xs, Y: batch_ys, dropout_rate:0.7})
    print ("cost:",cost.eval({X: batch_xs, Y: batch_ys, dropout_rate:0.7}))
  
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate:1}))
print(hypothesis.eval({X: mnist.test.images, Y: mnist.test.labels, dropout_rate:1}))

r = random.randint(0, mnist.test.num_examples -1)
print('Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print('Prediction:', sess.run(tf.argmax(hypothesis,1),{X:mnist.test.images[r:r+1], dropout_rate:1}))
print(mnist.test.images[r:r+1])

plt.imshow(mnist.test.images[r:r+1].reshape(28,28)
           , cmap='Greys', interpolation='nearest')
plt.show()

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import invert

img = imread('./data/9.jpg')
gray = rgb2gray(img).reshape(1,784)
gray = invert(gray)
print('Prediction:', sess.run(tf.argmax(hypothesis,1)
                              ,{X:gray, dropout_rate:1}))
plt.imshow(gray.reshape(28,28)
           , cmap='Greys', interpolation='nearest')
plt.show()