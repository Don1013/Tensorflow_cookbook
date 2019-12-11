import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# make data
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# make planning matrix A
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# make matrix b
b = np.transpose(np.matrix(y_vals))

# make tensors
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# cholesky decomposition
tA_A = tf.linalg.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.linalg.cholesky(tA_A)

# solve L * y = t(A)
tA_b = tf.linalg.matmul(tf.transpose(A_tensor), b)
sol1 = tf.linalg.solve(L, tA_b)

# solve L' * y = sol1
solution = tf.linalg.solve(tf.transpose(L), sol1)

# extract coefficients
slope = solution[0][0].numpy()
y_intercept = solution[1][0].numpy()
print('slope :' + str(slope))
print('y_intercept :' + str(y_intercept))

# get best fit line
best_fit = []
for i in x_vals:
  best_fit.append(slope * i + y_intercept)
  
# plot result
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()