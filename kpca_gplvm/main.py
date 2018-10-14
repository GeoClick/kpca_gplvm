import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from kpca_gplvm.util import convert_pca, convert_kpca, convert_gplvm, make_data_set

# Construct some artificial data set.
# Note that all methods are unsupervised. We will only use labels, Y, for visualisation
num_points = 100
X_train, y_train = make_data_set(num_points)
X_val, y_val = X_train, y_train
# X_val, y_val = make_data_set(200)


# # And plot the initial data in 3D
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train)
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.show()

# Apply normal PCA and plot
Z_pca_test = convert_pca(X_train, X_val)
fig, axarr = plt.subplots(2, 2)

axarr[0, 0].scatter(Z_pca_test[:, 0], Z_pca_test[:, 1], c=y_val)
axarr[0, 0].set_title('Normal PCA')
axarr[0, 0].set_xlabel('First component')
axarr[0, 0].set_ylabel('Second component')


# Apply kernel PCA and plot
Z_kpca_test = convert_kpca(X_train, X_val)

axarr[0, 1].scatter(Z_kpca_test[:, 0], Z_kpca_test[:, 1], c=y_val)
axarr[0, 1].set_title('kernel PCA')
axarr[0, 1].set_xlabel('First component')
axarr[0, 1].set_ylabel('Second component')

# Apply GP LVM and plot

Z_gplvm = convert_gplvm(X_train, X_val)

axarr[1, 0].scatter(Z_gplvm[:, 0], Z_gplvm[:, 1], c=y_train)
axarr[1, 0].set_title('gplvm')
axarr[1, 0].set_xlabel('First component')
axarr[1, 0].set_ylabel('Second component')
plt.show()
plt.waitforbuttonpress()
pass
a=0