import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
import os
import csv

# set name of data file
birth_weight_file = 'birth_weight.csv'

# download data anNo module named 'tensorflow.python.tools'; 'tensorflow.python' is not a packaged make data file
if not os.path.exists(birth_weight_file):
  birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
  birth_file = requests.get(birthdata_url)
  birth_data = birth_file.text.split('\r\n')
  birth_header = birth_data[0].split('\t')
  birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
  with open(birth_weight_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(birth_data)
    f.close()

# read data to memory
birth_data = []
with open(birth_weight_file, newline='') as csvfile:
  csv_reader = csv.reader(csvfile)
  birth_header = next(csv_reader)
  for row in csv_reader:
    birth_data.append(row)

birth_data = [[float(x) for x in row] for row in birth_data]
# extract objective variable
y_vals = np.array([x[0] for x in birth_data])
# extract explanatory variable
x_vals = np.array([x[2:9] for x in birth_data])

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
# normalize column
def normalize_cols(m):
  col_max = m.max(axis=0)
  col_min = m.min(axis=0)
  return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# define batch size, variable, and model
batch_size = 25
A = tf.Variable(tf.random.normal(shape=[7, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))
@tf.function
def model_output(x_data):
  return tf.add(tf.matmul(x_data, A), b)

# specified loss function
@tf.function
def loss(x_data, y_target):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output(x_data), labels=y_target))

# specified optimizer
  my_opt = tf.compat.v1.train.GradientDescentOptimizer(0.01)
@tf.function
def train_step(x_data, y_target):
  my_opt.minimize(loss(x_data, y_target))

# real prediction
@tf.function
def prediction(x_data):
  return tf.round(tf.sigmoid(model_output(x_data)))
@tf.function
def predictions_correct(x_data, y_terget):
  return tf.cast(tf.math.equal(prediction(x_data), y_target), tf.float32)
@tf.function
def accuracy(x_data, y_target):
  return tf.reduce_mean(predictions_correct(x_data, y_target))

# record loss value and accuracy
loss_vec = []
train_acc = []
test_acc = []
for i in range(1500):
  rand_index = np.random.choice(len(x_vals_train), size=batch_size)
  rand_x = x_vals_train[rand_index]
  rand_y = np.transpose([y_vals_train[rand_index]])
  x_data = tf.convert_to_tensor(rand_x, dtype='float32')
  y_target = tf.convert_to_tensor(rand_y, dtype='float32')
  print(x_data.shape)
  print(y_target.shape)
  x_vals_train_tensor = tf.convert_to_tensor(x_vals_train, dtype='float32')
  y_vals_train_tensor = tf.convert_to_tensor(np.transpose([y_vals_train]), dtype='float32')
  x_vals_test_tensor = tf.convert_to_tensor(x_vals_test, dtype='float32')
  y_vals_test_tensor = tf.convert_to_tensor(np.transpose([y_vals_test]), dtype='float32')
  print(x_vals_train_tensor.shape)
  print(y_vals_train_tensor.shape)
  print(x_vals_test_tensor.shape)
  print(y_vals_test_tensor.shape)
  temp_loss = loss(x_data, y_target)
  loss_vec.append(temp_loss)
  temp_acc_train = accuracy(x_vals_train_tensor, y_vals_train_tensor)
  train_acc.append(temp_acc_train)
  temp_acc_test = accuracy(x_vals_test_tensor, y_vals_test_tensor)
  test_acc.append(temp_acc_test)

# plot loss value
plt.plot(loss_vec, 'k-')
plt.title('Cross Entopy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# plot accuracy for train and test
plt.plot(train_acc, 'k--', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
