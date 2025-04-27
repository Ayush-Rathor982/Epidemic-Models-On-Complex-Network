# Import necessary libraries
import torch  # PyTorch library for building and training neural networks
import numpy as np  # NumPy for numerical operations
import networkx as nx  # NetworkX for network analysis (not used directly here though)
import matplotlib.pyplot as plt  # Matplotlib for plotting
from scipy.integrate import solve_ivp  # Solver for ODE systems
import pandas as pd  # Pandas for handling dataframes

# Load adjacency matrix (network structure) from file
A = np.loadtxt(r"matrix\ERmatrix50.txt", dtype=int)

# Number of nodes in the network
n = A.shape[0]
print(f"Number of nodes: {n}")

# SIS model parameters
beta = 0.0022  # Infection rate per contact
gamma = 0.03   # Recovery rate
t_max = 350    # Total simulation time

# Define the SIS model as a system of differential equations
def sis_dynamics(t, x):
    dxdt = np.zeros_like(x)  # Initialize derivative array
    for i in range(len(x)):
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)  # Infection term based on neighbors
        recovery_term = -gamma * x[i]  # Recovery term
        dxdt[i] = infection_sum + recovery_term  # Net change
    return dxdt

# Initial infection probabilities (small random values)
x0 = np.random.rand(n) * 0.01

# Solve the ODE using the RK45 method over 500 time points
solution = solve_ivp(
    sis_dynamics, [0, t_max], x0, t_eval=np.linspace(0, t_max, 500), method='RK45'
)

print(solution.y.shape)  # Shape: (num_nodes, time_points)

# Plot infection probabilities for each node over time
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(solution.t, solution.y[i], label=f'Node {i}', alpha=0.5)

plt.xlabel("Time")
plt.ylabel("Infection Probability (x_i)")
plt.title("SIS Model Dynamics")
plt.legend(loc='upper right', fontsize=6, ncol=2)
plt.grid()
plt.show()

# Convert solution to a time series matrix: rows = time points, columns = nodes
sis_time_series = solution.y.T
print(f"Generated SIS time series data shape: {sis_time_series.shape}")  # Should be (500, 50)

# Save the time series to a CSV file
# df = pd.DataFrame(sis_time_series)
# df.to_csv("sis_time_series.csv", index=False)

print("Saved as sis_time_series.csv")

# Function to prepare RNN training data using sliding window technique
def prepare_rnn_data(data, window_size=10):
    """
    Converts time series data into (X, Y) pairs using a sliding window.
    X: input sequences, Y: next-step prediction targets
    """
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Take window_size steps as input
        Y.append(data[i+window_size])    # Next step as target
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# Prepare input-output training pairs
window_size = 10
X, Y = prepare_rnn_data(sis_time_series, window_size)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")  # Expected: (495, 10, 50), (495, 50)

# Define the RNN model class
class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size  # Size of hidden layer
        self.num_layers = num_layers  # Number of RNN layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN Layer
        self.fc = torch.nn.Linear(hidden_size, output_size)  # Fully connected output layer

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initialize hidden state
        out, _ = self.rnn(x, h0)  # RNN forward pass
        out = self.fc(out[:, -1, :])  # Pass last time step output through FC layer
        return out

# RNN training hyperparameters
input_size = n  # Each node is a feature
hidden_size = 64  # Number of hidden units
output_size = n  # Predict next infection probability for each node
num_layers = 2  # Number of RNN layers
epochs = 2000  # Number of training epochs
lr = 0.01  # Learning rate
train_ratio = 0.5  # Fraction of data used for training

# Split data into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]  # Training data
X_test, Y_test = X[train_size:], Y[train_size:]    # Testing data

print("test ", X_test.shape, " texst ", Y_test.shape)

# Initialize the RNN model
model = RNNModel(input_size, hidden_size, output_size, num_layers)

# Loss function: Mean Squared Error (MSE)
criterion = torch.nn.MSELoss()

# Optimizer: Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# List to record loss at each epoch
losses = []

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    Y_pred_train = model(X_train)  # Forward pass
    loss = criterion(Y_pred_train, Y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    losses.append(loss.item())  # Save current loss

    if (epoch + 1) % 100 == 0:  # Print loss every 100 epochs
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing phase (no gradient tracking)
with torch.no_grad():
    Y_pred_test = model(X_test)  # Forward pass on test data
    test_loss = criterion(Y_pred_test, Y_test).item()  # Compute test loss
print(f"Test Loss: {test_loss}")

# Plot the training loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# Compute average infection probabilities across all nodes at each time step for train data
train_true_avg = Y_train.mean(dim=1).numpy()
train_pred_avg = Y_pred_train.mean(dim=1).detach().cpu().numpy()

# Compute average infection probabilities across all nodes at each time step for test data
true_avg = Y_test.mean(dim=1).numpy()
pred_avg = Y_pred_test.mean(dim=1).numpy()

print(train_true_avg.shape,"  ",train_pred_avg.shape)
print(true_avg.shape,"  ",pred_avg.shape)

# Plot true vs predicted average infection probability
plt.figure(figsize=(10, 5))

# Plot train true
plt.plot(train_true_avg, label='Train True', marker='o', linestyle='None', markersize=4, markerfacecolor='none', markeredgecolor='yellow')

# Plot train predicted
plt.plot(train_pred_avg, label='Train Predicted', marker='o', linestyle='None', markersize=1, markerfacecolor='none', markeredgecolor='red')

# Plot test true
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(true_avg)), true_avg, label='Test True Average')

# Plot test predicted
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(pred_avg)), pred_avg, label='Test Predicted Average', linestyle='dashed')

# Axis labels and title
plt.xlabel('Time Steps', fontsize=23, fontweight='bold')
plt.ylabel('Infection Probability', fontsize=23, fontweight='bold')
plt.title('RNN : True vs Predicted Infection Probability', fontsize=24, fontweight='bold')

# Adjust ticks font size
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')

# Show legend and plot
plt.legend()
plt.show()
