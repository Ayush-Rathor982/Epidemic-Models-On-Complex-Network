import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
beta = 0.3  # Infection rate constant
n = 1000    # Total population

# Initial conditions
S0 = 999      # Initial number of susceptible individuals
X0 = 1        # Initial number of infected individuals
t_span = (0, 100)  # Time range for the simulation
initial_conditions = [S0, X0]

# Differential equations for the SI model
def si_model(t, y, beta, n):
    S, X = y
    dS_dt = -beta * S * X / n  # Equation (17.2)
    dX_dt = beta * S * X / n   # Equation (17.1)
    return [dS_dt, dX_dt]

# Solving the differential equations
solution = solve_ivp(si_model, t_span, initial_conditions, args=(beta, n), dense_output=True)

# Extracting the results
t_values = np.linspace(t_span[0], t_span[1], 400)
S_values, X_values = solution.sol(t_values)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, S_values, label='Susceptible (S)')
plt.plot(t_values, X_values, label='Infected (X)')
plt.legend(fontsize=16)
plt.xlabel('Time(Days)',  fontsize=16)
plt.ylabel('Population',  fontsize=16)
plt.legend()
plt.title('SI Model: Susceptible and Infected Over Time', fontsize=17)
# plt.grid('false')
plt.show()
