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
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

matrix.reshape(2, 6)
matrix.reshape(-1, 1)
matrix.reshape(1, -1)
matrix.reshape(12)

# Transposing a vector or a matrix
matrix.T
np.array([[1, 2, 3, 4, 5, 6]]).T

# Flattening a matrix
matrix.flatten()
matrix.reshape(1, -1)

# Finding the rank of a matrix
np.linalg.matrix_rank(matrix)

# Calculating the determinant
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

np.linalg.det(matrix)

# Getting the diagonal of a matrix
matrix.diagonal()
matrix.diagonal(offset = 1)
matrix.diagonal(offset = -1)

# Calculating the trace of a matrix
sum(matrix.diagonal())
matrix.trace()

# Finding eigenvalues or eigenvectors
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

eigenvalues, eigenvectors = np.linalg.eig(matrix)
eigenvalues
eigenvectors

# Calculating dot products
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

np.dot(vector_a, vector_b)
vector_a @ matrix

# Adding and substracting matrices
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

np.add(matrix_a, matrix_b)
np.subtract(matrix_a, matrix_b)

matrix_a + matrix_b
matrix_a - matrix_b

# Multiplying matrices
np.dot(matrix_a, matrix_b)
matrix_a @ matrix_b

matrix_a * matrix_b

# Inverting a matrix
matrix = np.array([[1, 4, 2],
                   [2, 5, 1],
                   [2, 5, 3]])

np.linalg.inv(matrix)
matrix @ np.linalg.inv(matrix)

# Generating random values
np.random.seed(1)
np.random.random(3)
np.random.randint(0, 11, 3)
np.random.normal(0.0, 1.0, 3)
np.random.logistic(0.0, 2.0, 3)
np.random.uniform(1.0, 2.0, 3)



