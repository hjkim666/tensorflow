# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                    [[4],[5],[6]],
                    [[7],[8],[9]]                  
                  ]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
plt.show()

# In[4]:


print("image:\n", image)
weight = tf.constant([[[[1.]],[[1.]]]
                       ,[[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_img = conv2d.eval()
print(conv2d_img)
print(conv2d_img.shape)
 
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    plt.show()
# 
# 
# # In[6]:
# 
# 
# image = np.array([[[[1],[2],[3]],
#                     [[4],[5],[6]],
#                     [[7],[8],[9]]                  
#                   ]], dtype=np.float32)
# pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
#                     strides=[1, 1, 1, 1], padding='VALID')
# print(pool.shape)
# print(pool.eval())
# 
# 
# # In[7]:
# 
# 
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
print(pool.eval())

