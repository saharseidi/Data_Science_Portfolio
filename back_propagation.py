
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()


# In[2]:


sess = tf.Session()
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)


# In[3]:


# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))


# In[4]:


# Add operation to graph
my_output = tf.multiply(x_data, A)


# In[5]:


# Add L2 loss operation to graph
loss = tf.square(my_output - y_target)


# In[6]:



# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)


# In[7]:


# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)


# In[8]:


# Run Loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%25==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))


# In[9]:


ops.reset_default_graph()


# In[10]:



# Create graph
sess = tf.Session()


# In[11]:


# Create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)


# In[12]:


# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))


# In[13]:


# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
my_output = tf.add(x_data, A)

# Now we have to add another dimension to each (batch size of 1)
my_output_expanded = tf.expand_dims(my_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)


# In[14]:


xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)


# In[15]:


# Create Optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)


# In[17]:


# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)


# In[18]:


# Run loop
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i+1)%200==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))


# In[19]:


# Evaluate Predictions
predictions = []
for i in range(len(x_vals)):
    x_val = [x_vals[i]]
    prediction = sess.run(tf.round(tf.sigmoid(my_output)), feed_dict={x_data: x_val})
    predictions.append(prediction[0])
    
accuracy = sum(x==y for x,y in zip(predictions, y_vals))/100.
print('Ending Accuracy = ' + str(np.round(accuracy, 2)))

