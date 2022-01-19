
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops
ops.reset_default_graph()


# In[2]:


sess = tf.Session()

# Create data to feed in
my_array = np.array([[1., 3., 5., 7., 9.],
                   [-2., 0., 2., 4., 6.],
                   [-6., -3., 0., 3., 6.]])
# Duplicate the array for having two inputs
x_vals = np.array([my_array, my_array + 1])
# Declare the placeholder
x_data = tf.placeholder(tf.float32, shape=(3, 5))
# Declare constants for operations
m1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])


# In[3]:


# 1st Operation Layer = Multiplication
prod1 = tf.matmul(x_data, m1)


# In[4]:


# 2nd Operation Layer = Multiplication
prod2 = tf.matmul(prod1, m2)
# 3rd Operation Layer = Addition
add1 = tf.add(prod2, a1)


# In[5]:


for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

