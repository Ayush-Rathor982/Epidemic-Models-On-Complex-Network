import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# Parameters
beta = 0.4  # Transmission rate
gamma = 0.05  # Recovery rate
s0 = 0.99  # Initial fraction of susceptible individuals
x0 = 0.01  # Initial fraction of infected individuals
r0 = 0.0   # Initial fraction of recovered individuals

# Define the system of differential equations
def sir_model(t, y):
    s, x, r = y
    ds_dt = -beta * s * x
    dx_dt = beta * s * x - gamma * x
    dr_dt = gamma * x
    return [ds_dt, dx_dt, dr_dt]

# Time span for the simulation
time_span = (0, 160)  # Days
time_eval = np.linspace(time_span[0], time_span[1], 1000)

# Initial conditions
initial_conditions = [s0, x0, r0]

# Solve the differential equations
solution = solve_ivp(sir_model, time_span, initial_conditions, t_eval=time_eval, method='RK45')

# Extract solutions
s_values = solution.y[0]
x_values = solution.y[1]
r_values = solution.y[2]
t_values = solution.t

# Verify s + x + r = 1 at all times
conservation_check = np.allclose(s_values + x_values + r_values, 1.0)
print(f"Conservation of population fractions (s + x + r = 1): {conservation_check}")

# Compute the integral solution for r(t)
def integral_solution(u):
    return 1 / (1 - u - s0 * np.exp(-beta * u / gamma))

integral_values = []
for r in r_values:
    integral, _ = quad(integral_solution, 0, r)
    integral_values.append(integral / gamma)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, s_values, label='Susceptible (s)')
plt.plot(t_values, x_values, label='Infected (x)')
plt.plot(t_values, r_values, label='Recovered (r)')
plt.title('SIR Model Dynamics', fontsize=17)
plt.xlabel('Time (days)', fontsize=16)
plt.ylabel('Fraction of Population', fontsize=16)
plt.legend()
# plt.grid()
plt.show()


