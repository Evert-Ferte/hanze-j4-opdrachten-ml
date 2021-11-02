import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# ==== OPGAVE 1 ====
def plot_number(nrVector):
    grayScalePic = np.reshape(nrVector.T, (20, 20), 'F')

    plt.matshow(grayScalePic)
    plt.show()


# ==== OPGAVE 2a ====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ==== OPGAVE 2b ====
def get_y_matrix(y, m):
    cols = [e[0] if e[0] != 10 else 0 for e in y]  # TODO numpy.where
    rows = [i for i in range(m)]
    data = [1 for _ in range(m)]
    width = np.max(y)
    y_mat = csr_matrix((data, (rows, cols)), shape=(len(rows), width)).toarray()
    # print(y_mat)  # print the y matrix

    return y_mat


# ==== OPGAVE 2c ====
# ===== deel 1: =====
def predict_number(Theta1, Theta2, X):
    # Stap 1 - Input layer (L1)
    a1 = np.c_[np.ones(X.shape[0]), X]  # Voeg enen toe op index 0

    # Stap 2 - Hidden layer (L2)
    z2 = np.dot(Theta1, a1.T)  # De sommatie van Theta1 a1
    a2 = sigmoid(z2)  # De activatie (sigmoid) van z2
    a2 = np.c_[np.ones(a2.shape[1]), a2.T]

    # Stap 3 - Output layer (L3)
    z3 = np.dot(Theta2, a2.T)  # De sommatie van Theta2 a2
    a3 = sigmoid(z3)  # De activatie (sigmoid) van z3

    return a3.T

# ===== deel 2: =====
def compute_cost(Theta1, Theta2, X, y):
    m = y.shape[0]
    y_mat = get_y_matrix(y, m)
    h = predict_number(Theta1, Theta2, X)

    yLogH = y_mat * np.log(h)
    yLogHMin = (1 - y_mat) * np.log(1 - h)
    sumOfK = yLogH + yLogHMin
    sumOfM = np.sum(sumOfK)
    cost = -sumOfM / m

    # cost = -np.sum((y_mat * np.log(h)) + ((1 - y_mat) * np.log(1 - h))) / m

    return cost


# ==== OPGAVE 3a ====
def sigmoid_gradient(z):
    gZ = sigmoid(z)
    return (gZ * (1 - gZ))[0]


# ==== OPGAVE 3b ====
def nn_check_gradients(Theta1, Theta2, X, y):
    # Retourneer de gradiënten van Theta1 en Theta2, gegeven de waarden van X en van y
    # Zie het stappenplan in de opgaven voor een mogelijke uitwerking.

    Delta2 = np.zeros(Theta1.shape)
    Delta3 = np.zeros(Theta2.shape)
    m = X.shape[0]

    # region forward propagation

    # Stap 1 - Input layer (L1)
    a1 = np.c_[np.ones(X.shape[0]), X]  # Voeg enen toe op index 0

    # Stap 2 - Hidden layer (L2)
    z2 = np.dot(Theta1, a1.T)  # De sommatie van Theta1 a1
    a2 = sigmoid(z2)  # De activatie (sigmoid) van z2
    a2 = np.c_[np.ones(a2.shape[1]), a2.T]

    # Stap 3 - Output layer (L3)
    z3 = np.dot(Theta2, a2.T)  # De sommatie van Theta2 a2
    a3 = sigmoid(z3).T  # De activatie (sigmoid) van z3

    # endregion

    # region backward propagation

    for i in range(m):
        deltaL3 = a3[i] - y[i]
        deltaL2 = np.dot(deltaL3, Theta2) * sigmoid_gradient(z2.T[i])
        print("deltaL2, deltaL3, Theta2", deltaL2.shape, deltaL3.shape, Theta2.shape)

        rA2 = np.reshape(a2[i], (a2[i].shape[0], 1))
        deltaL3 = np.reshape(deltaL3, (deltaL3.shape[0], 1))
        rA1 = np.reshape(a1[i], (a1[i].shape[0], 1))
        print("rA1 shape", rA1.shape)
        deltaL2 = np.reshape(deltaL2, (deltaL2.shape[0], 1))

        # Theta2 shape    = (10, 26)
        # dot(delta3, a2) = (10, 26)
        # Theta1 shape    = (25, 401
        # dot(delta2, a1) = (26, 401)

        Theta2 = Theta2 + np.dot(deltaL3, rA2.T)  # reshape and T
        Theta1 = Theta1 + np.dot(deltaL2, rA1.T)

    Delta2_grad = Delta2 / m
    Delta3_grad = Delta3 / m

    # endregion

    return Delta2_grad, Delta3_grad
