import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


 
# Read the matrix from the file
A = np.loadtxt("Classical_Models/matrix/ERmatrix10.txt", dtype=int)

# Print the matrix
print("Matrix read from file:")
print(A)


n = A.shape[0]  # Number of nodes
print(f"Number of nodes in the matrix: {n}")



# Define parameters
beta = 0.3  # Infection rate
gamma = 0.01  # Recovery rate
t_max = 100  # Maximum time for simulation



# Eigenvector centrality for initialization
eigenvalues, eigenvectors = np.linalg.eig(A)
kappa1 = max(eigenvalues)  # Leading eigenvalue

def sis_dynamics(t, x):
    """
    Differential equation for the SIS model.
    """
    dxdt = np.zeros_like(x)
    for i in range(len(x)):
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)
        recovery_term = -gamma * x[i]
        dxdt[i] = infection_sum + recovery_term
    return dxdt

# Initial condition
x0 = np.random.rand(n) * 0.01  # Small random initial infection probabilities

# Solve the differential equation
solution = solve_ivp(
    sis_dynamics, [0, t_max], x0, t_eval=np.linspace(0, t_max, 500), method='RK45'
)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(solution.t, solution.y[i], label=f'Node {i}', alpha=0.5)

plt.xlabel("Time")
plt.ylabel("Infection Probability (x_i)")
plt.title("SIS Model Dynamics")
plt.legend(loc='upper right', fontsize=6, ncol=2)
plt.grid()
plt.show()

# Analyze long-term behavior
steady_state = solution.y[:, -1]
print("Steady-state infection probabilities:")
for i, prob in enumerate(steady_state):
    print(f"Node {i}: {prob:.4f}")






from collections import defaultdict

# Compute degrees of nodes
degrees = np.sum(A, axis=1)

# Group nodes by degree
degree_groups = defaultdict(list)
for node, degree in enumerate(degrees):
    degree_groups[degree].append(node)

# Compute degree-wise averages over time
degree_avg_dynamics = {}
for degree, nodes in degree_groups.items():
    degree_avg_dynamics[degree] = np.mean(solution.y[nodes, :], axis=0)

# Plot degree-wise dynamics
plt.figure(figsize=(10, 6))

for degree, avg_dynamics in sorted(degree_avg_dynamics.items()):
    plt.plot(solution.t, avg_dynamics, label=f'Degree {degree}')

plt.xlabel("Time")
plt.ylabel("Average Infection Probability")
plt.title("SIS Model Dynamics Averaged by Degree")
plt.legend(title="Node Degree", loc='upper right', fontsize=8)
plt.xscale('log')
plt.grid()
plt.show()

# Analyze and print steady-state behavior degree-wise
print("Degree-wise steady-state infection probabilities:")
for degree, avg_dynamics in sorted(degree_avg_dynamics.items()):
    print(f"Degree {degree}: {avg_dynamics[-1]:.4f}")
