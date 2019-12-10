import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# make data
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
# make planning matrix A
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))
# make matrix b
b = np.transpose(np.matrix(y_vals))
# make tensor
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)
# apply inverse matrix method
tA_A = tf.linalg.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.linalg.inv(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.linalg.matmul(product, b_tensor)

solution_eval = solution

# extract coefficients
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope :' + str(slope.numpy()))
print('y_intercept :' + str(y_intercept.numpy()))

# get best line
best_fit = []
for i in x_vals:
  best_fit.append(slope * i + y_intercept)

# plot this results
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Beat fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()