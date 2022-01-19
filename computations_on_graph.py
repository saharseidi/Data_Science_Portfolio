
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# In[3]:


sess = tf.Session()
# Create data to feed in the placeholder
x_vals = np.array([1., 3., 5., 7., 9.])

# Create the TensorFlow Placceholder
x_data = tf.placeholder(tf.float32)

# Constant for multiplication
m = tf.constant(3.)
# Multiplication
prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))


# In[5]:


merged = tf.summary.merge_all(key='summaries')
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

my_writer = tf.summary.FileWriter('tensorboard_logs/', sess.graph)

