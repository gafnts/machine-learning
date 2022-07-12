import numpy as np
from scipy import sparse

# Vector as row
vector_row = np.array([1, 2, 3])

# Vector as column
vector_column = np.array([[1],
                          [2],
                          [3]])

# Matrix
matrix = np.array([[1, 2],
                   [2, 3],
                   [3, 4]])

# Not recommended
matrix_object = np.mat([[1, 2],
                        [2, 3],
                        [3, 4]])

# Sparse matrix
mat = np.array([[1, 0],
                [0, 1],
                [1, 0]])

matrix_sparse = sparse.csr_matrix(mat)
print(matrix_sparse)

# Larger matrix
mat_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

matrix_large_sparse = sparse.csr_matrix(mat_large)
print(matrix_large_sparse)

# Selecting elements
vector = np.array([1, 2, 3, 4, 5, 6])

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

vector[2]
vector[:]
vector[:3]
vector[3:]
vector[-1]

matrix[1, 1]
matrix[:2, :]
matrix[:, 0]
matrix[:, 1:2]

# Describing a matrix
matrix.shape
matrix.size
matrix.ndim

# Applying operations to elements
add_100 = lambda i: i + 100
vectorized_add_100 = np.vectorize(add_100)
vectorized_add_100(matrix)

# Broadcasting
matrix + 100

# Finding the maximum and minimum values
np.max(matrix)
np.min(matrix)

np.max(matrix, axis = 0) # Column
np.max(matrix, axis = 1) # Row

# Average, variance and standard deviation
np.mean(matrix)
np.var(matrix)
np.std(matrix)

np.mean(matrix, axis = 0) # Column
np.mean(matrix, axis = 1) # Row

# Reshaping arrays









































