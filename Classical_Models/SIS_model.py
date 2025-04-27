import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Differential equations for the SIS model
def sis_model(y, t, beta, gamma):
    s, x = y  # s: susceptible fraction, x: infected fraction
    ds_dt = gamma * x - beta * s * x
    dx_dt = beta * s * x - gamma * x
    return [ds_dt, dx_dt]

# Parameters
beta = 0.4  # Infection rate
gamma = 0.1  # Recovery rate

# Initial conditions
x0 = 0.01  # Initial infected fraction
s0 = 1 - x0  # Initial susceptible fraction

# Time points
time = np.linspace(0, 100, 1000)

# Solve the SIS model
y0 = [s0, x0]  # Initial conditions
dynamics = odeint(sis_model, y0, time, args=(beta, gamma))

# Extract results
s = dynamics[:, 0]
x = dynamics[:, 1]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, s, label='Susceptible (s)', color='blue')
plt.plot(time, x, label='Infected (x)', color='red')
plt.axhline((beta - gamma) / beta, color='green', linestyle='--', label='Endemic state')
plt.title('SIS Model Dynamics', fontsize=16)
plt.xlabel('Time(Days)', fontsize=16)
plt.ylabel('Fraction of Population', fontsize=17)
plt.legend()
# plt.grid()
plt.show()




