import numpy as np
import networkx as nx

# Parameters for the Scale-Free Graph
n = 50  # Number of nodes
m = 2   # Number of edges to attach from a new node to existing nodes

# Generate the Scale-Free Graph using the Barabási–Albert model
sf_graph = nx.barabasi_albert_graph(n, m)

# Generate the adjacency matrix
sf_adj_matrix = nx.adjacency_matrix(sf_graph).toarray()

# Save the matrix to a file
np.savetxt("SFmatrix50.txt", sf_adj_matrix, fmt="%d")

# Print the adjacency matrix
print("Scale-Free Adjacency Matrix:")
print(sf_adj_matrix)
