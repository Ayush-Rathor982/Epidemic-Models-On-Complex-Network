import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the SIR model equations
def sir_model(t, y, beta, gamma, A):
    n = len(A)
    s = y[:n]  # Susceptible
    x = y[n:2*n]  # Infected
    r = y[2*n:]  # Recovered
    
    print("y ", y)

    ds_dt = -beta * s * (A @ x)
    dx_dt = beta * s * (A @ x) - gamma * x
    dr_dt = gamma * x

    return np.concatenate([ds_dt, dx_dt, dr_dt])


# Read the matrix from the file
A = np.loadtxt("Classical_Models/matrix/ERmatrix10.txt", dtype=int)

# Print the matrix
print("Matrix read from file:")
print(A)


n = A.shape[0]  # Number of nodes
print(f"Number of nodes in the matrix: {n}")



# Parameters
beta = 0.15  # Infection rate
gamma = 0.07  # Recovery rate
# mean_degree = 4  # Mean degree for Poisson degree distribution



# Initial conditions
c = 2  # Initial number of infected individuals
s_initial = np.ones(n) - c / n
x_initial = np.zeros(n)
x_initial[:c] = c / n
r_initial = np.zeros(n)
initial_conditions = np.concatenate([s_initial, x_initial, r_initial])

# Time span for the simulation
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the differential equations
solution = solve_ivp(
    sir_model,
    t_span,
    initial_conditions,
    args=(beta, gamma, A),
    t_eval=t_eval,
    method='RK45'
)

# Extract the results
s = solution.y[:n, :].mean(axis=0)  # Average susceptible fraction
x = solution.y[n:2*n, :].mean(axis=0)  # Average infected fraction
r = solution.y[2*n:, :].mean(axis=0)  # Average recovered fraction

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, s, label='Susceptible', color='blue')
plt.plot(solution.t, x, label='Infected', color='red')
plt.plot(solution.t, r, label='Recovered', color='green')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SIR Model on a Network')
plt.legend()
# plt.grid()
plt.show()





from collections import defaultdict

# Compute degrees of nodes
degrees = np.sum(A, axis=1)

# Group nodes by degree
degree_groups = defaultdict(list)
for node, degree in enumerate(degrees):
    degree_groups[degree].append(node)

# Extract the results for susceptible, infected, and recovered
s_all = solution.y[:n, :]  # Susceptible
x_all = solution.y[n:2*n, :]  # Infected
r_all = solution.y[2*n:, :]  # Recovered

# Compute degree-wise averages
s_avg_by_degree = {}
x_avg_by_degree = {}
r_avg_by_degree = {}

for degree, nodes in degree_groups.items():
    s_avg_by_degree[degree] = np.mean(s_all[nodes, :], axis=0)
    x_avg_by_degree[degree] = np.mean(x_all[nodes, :], axis=0)
    r_avg_by_degree[degree] = np.mean(r_all[nodes, :], axis=0)

# Plot the degree-wise averaged results
fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

# Plot degree-wise susceptible fractions
for degree, s_avg in sorted(s_avg_by_degree.items()):
    ax.plot(solution.t, s_avg, label=f'Degree {degree}')


# Plot degree-wise infected fractions
for degree, x_avg in sorted(x_avg_by_degree.items()):
    ax.plot(solution.t, x_avg, label=f'Degree {degree}')


# Plot degree-wise recovered fractions
for degree, r_avg in sorted(r_avg_by_degree.items()):
    ax.plot(solution.t, r_avg, label=f'Degree {degree}')
ax.set_xlabel('Time',fontsize=16)
ax.set_ylabel('Probability',fontsize=16)
ax.set_title('Infection Probabilities: SIR Model', fontsize=17)


# Adjust layout and show the plot
plt.tight_layout()
plt.show()

























