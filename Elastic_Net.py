import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()

# iris.data = [Sepal Length, Sepal Width, Petal Length, Petal Width]
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# define batch size, variables, and model
batch_size = 50
learning_rate = 0.001

A = tf.Variable(tf.random.normal(shape=[3, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))

@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

elastic_param1 = tf.constant(1.0)
elastic_param2 = tf.constant(1.0)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.multiply(elastic_param1, l1_a_loss)
e2_term = tf.multiply(elastic_param2, l2_a_loss)
@tf.function
def loss(x_data, y_target):
  return tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output(x_data))), e1_term), e2_term), 0)

# specified optimizer
my_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
@tf.function
def train_step(x_data, y_target):
  my_opt.minimize(loss(x_data, y_target), var_list=[A, b])

# start train step
loss_vec = []
for i in range(1000):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = x_vals[rand_index]
  rand_y = np.transpose(y_vals[rand_index])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step(x_data, y_target)
  temp_loss = loss(x_data, y_target)
  loss_vec.append(temp_loss[0])
  if (i + 1) % 250 == 0:
    print('Step # ' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    print('Loss = ' + str(temp_loss.numpy()))

# extract coeficient
[[sw_coef], [pl_coef], [pw_coef]] = A.numpy()
[y_intercept] = b.numpy()

# plot loss value
plt.plot(loss_vec, 'k-')
plt.title('Loss per Genaration')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()