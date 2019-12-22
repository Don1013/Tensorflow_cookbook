import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# L1 loss function
# load iris dataset
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# settings
batch_size = 25
learning_rate = 0.1
iterations = 50

# define variables and model
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))
@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

# change loss function to l1 loss function
@tf.function
def loss_l1(x_data, y_target):
  return tf.reduce_mean(tf.abs(y_target - model_output(x_data)))

# specified optimizer
@tf.function
def train_step_l1(x_data, y_target):
  my_opt_l1 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
  my_opt_l1.minimize(loss_l1(x_data, y_target), var_list=[A, b])

# start trainning loop
loss_vec_l1 = []
for i in range(iterations):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step_l1(x_data, y_target)
  temp_loss_l1 = loss_l1(x_data, y_target)
  loss_vec_l1.append(temp_loss_l1)
  if (i + 1) % 25 == 0:
    print('Step #' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))


# L2 loss function
# define variables and model
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))
@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

# change loss function to l2 loss function
@tf.function
def loss_l2(x_data, y_target):
  return tf.reduce_mean(tf.square(y_target - model_output(x_data)))

# specified optimizer
@tf.function
def train_step_l2(x_data, y_target):
  my_opt_l2 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
  my_opt_l2.minimize(loss_l2(x_data, y_target), var_list=[A, b])

# start trainning loop
loss_vec_l2 = []
for i in range(iterations):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step_l2(x_data, y_target)
  temp_loss_l2 = loss_l2(x_data, y_target)
  loss_vec_l2.append(temp_loss_l2)
  if (i + 1) % 25 == 0:
    print('Step #' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    
# plot graph
plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
plt.plot(loss_vec_l2, 'r--', label='L2 Loss')
plt.title('L1 and L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L1 Loss')
plt.legend(loc='upper right')
plt.show()