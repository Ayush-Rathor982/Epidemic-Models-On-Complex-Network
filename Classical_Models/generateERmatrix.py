import numpy as np

def generate_er_graph(n, p):
    """
    Generates an adjacency matrix for an Erdős–Rényi graph G(n, p).

    Parameters:
        n (int): Number of nodes.
        p (float): Probability of edge creation.

    Returns:
        np.ndarray: Adjacency matrix of the graph.
    """
    # Create an empty adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)

    # Generate edges based on the probability p
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() <= p:
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1  # Ensure symmetry for an undirected graph
    
    return adj_matrix

def save_matrix_to_file(matrix, filename):
    """
    Saves the adjacency matrix to a file.

    Parameters:
        matrix (np.ndarray): Adjacency matrix to save.
        filename (str): Name of the file to save the matrix.
    """
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

# Parameters
n = 500  # Number of nodes
p = 0.5  # Probability of edge creation

# Generate the ER graph adjacency matrix
adj_matrix = generate_er_graph(n, p)

# Save the adjacency matrix to a file
output_file = "matrix/prob/ER1.0matrix500.txt"
save_matrix_to_file(adj_matrix, output_file)

print(f"Adjacency matrix for G({n}, {p}) generated and saved to {output_file}.")
