import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# define batch size , variables, and model
batch_size = 50
learning_rate = 0.001
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))
@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

# define LASSO loss function
LASSO_param = tf.constant(0.9)
heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, LASSO_param)))))
regularization_param = tf.multiply(heavyside_step, 99.)
@tf.function
def loss_LASSO(x_data, y_target):
  return tf.add(tf.reduce_mean(tf.square(y_target - model_output(x_data))), regularization_param)

# specified optimizer
my_opt_LASSO = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
@tf.function
def train_step_LASSO(x_data, y_target):
  my_opt_LASSO.minimize(loss_LASSO(x_data, y_target), var_list=[A, b])

loss_vec_LASSO = []
print('LASSO')
for i in range(1500):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step_LASSO(x_data, y_target)
  temp_loss_LASSO = loss_LASSO(x_data, y_target)
  loss_vec_LASSO.append(temp_loss_LASSO[0])
  if (i + 1) % 300 == 0:
    print('Step # ' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    print('Loss = ' + str(temp_loss_LASSO.numpy()))
print('')

# define ridge loss function
ridge_param = tf.constant(1.0)
ridge_loss = tf.reduce_mean(tf.square(A))
@tf.function
def loss_ridge(x_data, y_target):
  return tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target - model_output(x_data))), tf.multiply(ridge_param, ridge_loss)), 0)

# specified optimizer
my_opt_ridge = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
@tf.function
def train_step_ridge(x_data, y_target):
  my_opt_ridge.minimize(loss_ridge(x_data, y_target), var_list=[A, b])

loss_vec_ridge = []
print('ridge')
for i in range(1500):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step_ridge(x_data, y_target)
  temp_loss_ridge = loss_ridge(x_data, y_target)
  loss_vec_ridge.append(temp_loss_ridge)
  if (i + 1) % 300 == 0:
    print('Step # ' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    print('Loss = ' + str(temp_loss_ridge.numpy()))

plt.plot(loss_vec_LASSO, 'k-', label='LASSO')
plt.plot(loss_vec_ridge, 'r--', label='ridge')
plt.title('LASSO and ridge per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()