import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.mlab as mlab


def draw_graph(data):
    x, y = data.T
    plt.scatter(x, y)
    plt.show()


def compute_cost(X, y, theta):
    return np.sum(np.square(np.dot(X, theta) - y)) / (2 * y.shape[0])


def gradient_descent(X, y, theta, alpha, num_iters):
    m, n = X.shape
    costs = []

    for _ in range(num_iters):
        for j in range(n):  # TODO - Haal deze lus eruit, maak factorieel
            # In steps:
            # h = (np.dot(X, theta.T) - y)
            # xj = X[:, [j]]
            # sum = np.sum(h * xj)
            # sumDivM = (sum / m)
            # theta[0][j] = theta[0][j] - alpha * (sumDivM)

            theta[0][j] = theta[0][j] - alpha * (np.sum((np.dot(X, theta.T) - y) * X[:, [j]]) / m)
        costs.append(compute_cost(X, y, theta.T))
    return theta, costs


def draw_costs(data):
    plt.plot(data)
    plt.show()


def contour_plot(X, y):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    jet = plt.get_cmap('jet')

    t1 = np.linspace(-10, 10, 100)
    t2 = np.linspace(-1, 4, 100)
    T1, T2 = np.meshgrid(t1, t2)

    J_vals = np.zeros((len(t2), len(t2)))

    # YOUR CODE HERE
    for i in range(len(t1)):
        theta1 = t1[i]
        for j in range(len(t2)):
            theta2 = t2[j]
            J_vals[i][j] = compute_cost(X, y, np.array((theta1, theta2)).T)

    surf = ax.plot_surface(T1, T2, J_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    xLabel = ax.set_xlabel(r'$\theta_0$', linespacing=3.2)
    yLabel = ax.set_ylabel(r'$\theta_1$', linespacing=3.1)
    zLabel = ax.set_zlabel(r'$J(\theta_0, \theta_1)$', linespacing=3.4)

    ax.dist = 10

    plt.show()
