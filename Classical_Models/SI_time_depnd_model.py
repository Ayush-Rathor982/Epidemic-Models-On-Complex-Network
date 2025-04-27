
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig
import matplotlib.pyplot as plt


# Read the matrix from the file
A = np.loadtxt("Classical_Models/matrix/ERmatrix5.txt", dtype=int)

# Print the matrix
print("Matrix read from file:")
print(A)


n = A.shape[0]
print(f"Number of nodes in the matrix: {n}")



# Define the differential equation
def si_model(t, x, beta, A):
    dxdt = beta * A @ x
    return dxdt



beta = 0.1  # Infection rate


# Initial condition: small infection at one vertex
x0 = np.zeros(n)
x0[0] = 0.5  # Vertex 0 is initially infected

# Time range for simulation
t_span = (0, 10)  # From t=0 to t=50
t_eval = np.linspace(*t_span, 100)  # 500 time points for evaluation

# Solve the differential equation
solution = solve_ivp(si_model, t_span, x0, args=(beta, A), t_eval=t_eval, method='RK45')

# Extract numerical solution
t = solution.t
x_numeric = solution.y

# Eigen decomposition of the adjacency matrix
eigenvalues, eigenvectors = eig(A)

print("eigenvalues ", eigenvalues)
print("eigenvectors ", eigenvectors)

# Identify the largest eigenvalue and its corresponding eigenvector
max_eigenvalue_idx = np.argmax(eigenvalues.real)
kappa1 = eigenvalues[max_eigenvalue_idx].real
v1 = eigenvectors[:, max_eigenvalue_idx].real

print("ind ",max_eigenvalue_idx, " kappa ", kappa1, " v1 ", v1)

# Normalize the eigenvector
v1 = v1 / np.linalg.norm(v1)


# Calculate x(t) using the eigenvector approximation
a0 = np.dot(v1, x0)  # Initial coefficient a_r(0)
x_approx = np.array([a0 * np.exp(beta * kappa1 * time) * v1 for time in t]).T


# Create a single figure with one subplot
fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

# Plot the numerical results
for i in range(n):
    ax.plot(t, x_numeric[i], label=f'Numerical: Vertex {i}')
    ax.plot(t, x_approx[i], marker='o', linestyle='None', label=f'Approximation: Vertex {i}')

# Customize the plot
ax.set_title("Infection Probabilities: Numerical Solution", fontsize=16)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Infection Probability", fontsize=14)
ax.legend()
ax.grid()


# Adjust layout for better spacing
plt.tight_layout()

# Show the combined plot
plt.show()










# Compute degrees of nodes
degrees = np.sum(A, axis=1)

# Group nodes by degree
from collections import defaultdict

degree_groups = defaultdict(list)
for node, degree in enumerate(degrees):
    degree_groups[degree].append(node)

# Compute the average infection probabilities for each degree group
x_avg_by_degree = {}
for degree, nodes in degree_groups.items():
    x_avg_by_degree[degree] = np.mean(x_numeric[nodes, :], axis=0)

# Plot the averaged results
fig, ax = plt.subplots(figsize=(12, 8))

for degree, x_avg in sorted(x_avg_by_degree.items()):
    ax.plot(t, x_avg, label=f'Degree {degree} (Avg)')

# Customize the plot
ax.set_title("Infection Probabilities Averaged by Degree", fontsize=16)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Average Infection Probability", fontsize=14)
ax.set_yscale('log')
# ax.set_xscale('log')
ax.legend()
ax.grid()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()













# print("a0 ",a0)
# print("x_app ",x_approx)

# # Plot the results
# plt.figure(figsize=(12, 8))
# for i in range(n):
#     plt.plot(t, x_numeric[i], label=f'Numerical: Vertex {i}')
#     # plt.plot(t, x_approx[i], '--', label=f'Approximation: Vertex {i}')

# plt.title("Infection Probabilities: Numerical vs Approximation", fontsize=16)
# plt.xlabel("Time", fontsize=14)
# plt.ylabel("Infection Probability", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()



# # Plot the results
# plt.figure(figsize=(12, 8))
# for i in range(n):
#     # plt.plot(t, x_numeric[i], label=f'Numerical: Vertex {i}')
#     plt.plot(t, x_approx[i], '--', label=f'Approximation: Vertex {i}')

# plt.title("Infection Probabilities: Numerical vs Approximation", fontsize=16)
# plt.xlabel("Time", fontsize=14)
# plt.ylabel("Infection Probability", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()


# # Create a single figure with two subplots
# fig, axs = plt.subplots(1, 1, figsize=(12, 6), sharex=True)

# # Plot the numerical results
# for i in range(n):
#     axs[0].plot(t, x_numeric[i], label=f'Numerical: Vertex {i}')
#     # axs[0].plot(t, x_approx[i], '--', label=f'Approximation: Vertex {i}')
#     axs[0].plot(t, x_approx[i], marker='o', linestyle='None', label=f'Approximation: Vertex {i}')


# axs[0].set_title("Infection Probabilities: Numerical Solution", fontsize=16)
# axs[0].set_ylabel("Infection Probability", fontsize=14)
# axs[0].legend()
# axs[0].grid()

# # Plot the approximation results
# for i in range(n):
#     print("ind ",i, "val ",x_approx[i])
#     axs[1].plot(t, x_approx[i], '--', label=f'Approximation: Vertex {i}')

# axs[1].set_title("Infection Probabilities: Approximation Solution", fontsize=16)
# axs[1].set_xlabel("Time", fontsize=14)
# axs[1].set_ylabel("Infection Probability", fontsize=14)
# axs[1].legend()
# axs[1].grid()











# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Define the differential equation
# def si_model(t, x, beta, A):
#     dxdt = beta * A @ x
#     return dxdt

# # Parameters
# n = 5  # Number of vertices
# beta = 0.2  # Infection rate

# # Example adjacency matrix (undirected graph)
# A = np.array([
#     [0, 1, 0, 0, 1],
#     [1, 0, 1, 0, 1],
#     [0, 1, 0, 1, 1],
#     [0, 0, 1, 0, 1],
#     [1, 1, 1, 1, 0],
# ])

# # Initial condition: small infection at one vertex
# x0 = np.zeros(n)
# x0[0] = 0.5  # Vertex 0 is initially infected

# # Time range for simulation
# t_span = (0, 3)  # From t=0 to t=30
# t_eval = np.linspace(*t_span, 20)  # 500 time points for evaluation

# # Solve the differential equation
# solution = solve_ivp(si_model, t_span, x0, args=(beta, A), t_eval=t_eval, method='RK45')

# # Extract results
# t = solution.t
# x = solution.y

# # Create the figure and axis for the bar plot
# fig, ax = plt.subplots(figsize=(10, 6))
# bars = ax.bar(range(n), x[:, 0], color='skyblue')

# # Configure the plot
# ax.set_title("Infection Probabilities Over Time", fontsize=16)
# ax.set_xlabel("Vertices", fontsize=14)
# ax.set_ylabel("Infection Probability", fontsize=14)
# ax.set_ylim(0, 1)
# ax.set_xticks(range(n))
# ax.set_xticklabels([f"Vertex {i}" for i in range(n)])
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Update function for animation
# def update(frame):
#     for bar, h in zip(bars, x[:, frame]):
#         bar.set_height(h)
#     ax.set_title(f"Infection Probabilities at Time {t[frame]:.2f}", fontsize=16)

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(t), interval=50, repeat=False)

# # Show the animation
# plt.show()

