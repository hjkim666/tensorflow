import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
tf.reset_default_graph()
tf.set_random_seed(107)

train_ecg = pd.read_csv("ecg_discord_train.csv", header=None)
test_ecg = pd.read_csv("ecg_discord_test.csv", header=None)

print(train_ecg.shape)
print(train_ecg.head())
print(test_ecg.shape)
print(test_ecg.head())

learning_rate = 0.0001
training_epochs = 13000
display_step = 1
examples_to_show = 10

n_input = 210
n_hidden_1 = 50 # 첫번째 층의 뉴런(특징, 속성) 갯수
n_hidden_2 = 20 # 두번째 층의 뉴런(특징, 속성) 갯수

X = tf.placeholder("float", [None, n_input])

# 인코더 생성
EW1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
Eb1 = tf.Variable(tf.random_normal([n_hidden_1]))
EL1 = tf.nn.tanh(tf.matmul(X, EW1) + Eb1)

EW2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
Eb2 = tf.Variable(tf.random_normal([n_hidden_2]))
EL2 = tf.nn.tanh(tf.matmul(EL1, EW2) + Eb2)

# 디코더 생성
DW1 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
Db1 = tf.Variable(tf.random_normal([n_hidden_1]))
DL1 = tf.nn.tanh(tf.matmul(EL2, DW1) + Db1)

DW2 = tf.Variable(tf.random_normal([n_hidden_1, n_input]))
Db2 = tf.Variable(tf.random_normal([n_input]))
DL2 = tf.matmul(DL1, DW2) + Db2

y_pred = DL2
y_true = X

cost = tf.reduce_mean(tf.square(y_true-y_pred))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    import time
    start = time.time()
    for epoch in range(training_epochs):
        _, c = sess.run([train_step, cost], feed_dict={X: train_ecg})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("훈련 시간:", time.time() - start)  

## 모델 평가
    test_recon = sess.run(y_pred, feed_dict={X: test_ecg})

print(test_recon[0:5, 0:5])
print(test_ecg.iloc[0:5, 0:5])
test_recon_error = ((test_recon - test_ecg) ** 2).mean(axis=1)
print(test_recon_error)
print("평균 복원 오차:", np.mean(test_recon_error))

# 이상 데이터 (ouliers) (마지막 3개의 ECG 데이터)를 시각화
#plt.ion()
plt.plot(test_recon_error)
plt.show()
