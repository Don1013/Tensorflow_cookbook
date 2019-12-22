import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# load Iris data set
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])  # petal width
y_vals = np.array([y[0] for y in iris.data])  # sepal length

# set learning rate and batch size
learning_rate = 0.05
batch_size = 25

# placeholder is deleted in tensorflow2.0
# so this code is written without placeholder 

# make variable
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))

# set L2 loss function
@tf.function
def loss(x_data, y_target):
  model_output = tf.add(tf.matmul(x_data, A), b)
  return tf.math.reduce_mean(tf.square(y_target - model_output))

# set optimizer
@tf.function
def train_step(x_data, y_taeget):
  my_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
  my_opt.minimize(loss(x_data, y_target), var_list=[A, b])

loss_vec = []
for i in range(100):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step(x_data, y_target)
  temp_loss = loss(x_data, y_target)
  loss_vec.append(temp_loss)
  if (i + 1) % 25 == 0:
    print('Step #' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    print('Loss = ' + str(temp_loss.numpy()))

# extract coefficients
[slope] = A.numpy()
[y_intercept] = b.numpy()

# get best fit line
best_fit = []
for i in x_vals:
  best_fit.append(slope * i + y_intercept)
  
# make graph
#1st graph
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Sepal Length')
plt.show()

#2nd graph
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()