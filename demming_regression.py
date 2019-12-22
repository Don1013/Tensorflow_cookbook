import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# load ieis dataset
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# define batchsize, Variables and model
batch_size = 50
A = tf.Variable(tf.random.normal(shape=[1, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))
@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

# define demming loss function
@tf.function
def demming_numerator(x_data, y_target):
  return tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
@tf.function
def loss(x_data, y_target):
  return tf.reduce_mean(tf.truediv(demming_numerator(x_data, y_target), demming_denominator))

# specified optimizer
my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.25)
@tf.function
def train_step(x_data, y_target):
  my_opt.minimize(loss(x_data, y_target), var_list=[A, b])

loss_vec = []
for i in range(1500):
  rand_index = np.random.choice(len(x_vals), size=batch_size)
  rand_x = np.transpose([x_vals[rand_index]])
  rand_y = np.transpose([y_vals[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  train_step(x_data, y_target)
  temp_loss = loss(x_data, y_target)
  loss_vec.append(temp_loss)
  if (i + 1) % 100 == 0:
    print('Step #' + str(i + 1) + ' A = ' + str(A.numpy()) + ' b = ' + str(b.numpy()))
    print('Loss = ' + str(temp_loss))

# extract coefficient
[slope] = A.numpy()
[y_intercept] = b.numpy()

# get best fitting line
best_fit = []
for i in x_vals:
  best_fit.append(slope * i + y_intercept)
  
# plot
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()