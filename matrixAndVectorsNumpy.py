import numpy as np

matrix = np.random.normal(loc=1, scale=10, size=(50, 1000))

average = np.mean(matrix, axis=0)
st_deviation = np.std(matrix, axis=0)

norma = (matrix - average) / st_deviation

Z = np.array([[4, 5, 0],
             [1, 9, 3],
             [5, 1, 1],
             [3, 3, 3],
             [9, 9, 9],
             [4, 7, 1]])

sum = np.sum(Z, axis=1)

E1 = np.eye(3)
E2 = np.eye(3)

print(np.vstack((E1, E2)))