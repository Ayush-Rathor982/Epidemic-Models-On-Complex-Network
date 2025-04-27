import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import networkx as nx
from scipy.stats import pearsonr

# Generate Erdős-Rényi network
n = 400  # Number of nodes
p = 0.01  # Probability for edge creation# Import necessary libraries
import torch  # For deep learning
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
from scipy.integrate import solve_ivp  # For solving differential equations
import networkx as nx  # For generating networks
from scipy.stats import pearsonr  # For calculating Pearson correlation coefficient

# Generate an Erdős-Rényi random network
n = 400  # Number of nodes
p = 0.01  # Probability for edge creation
G = nx.erdos_renyi_graph(n, p)  # Create ER random graph
A = nx.adjacency_matrix(G).toarray()  # Get adjacency matrix as NumPy array
print(f"Number of nodes: {n}")

# SIS (Susceptible-Infected-Susceptible) Model Parameters
beta = 0.012  # Infection rate
gamma = 0.03  # Recovery rate
t_max = 350  # Total time for simulation

# Define SIS model differential equations
def sis_dynamics(t, x):
    dxdt = np.zeros_like(x)  # Initialize dx/dt array
    for i in range(len(x)):  # For each node
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)  # Infection from neighbors
        recovery_term = -gamma * x[i]  # Recovery term
        dxdt[i] = infection_sum + recovery_term  # Total change for node i
    return dxdt

# Set initial infection probabilities (very small random infections)
x0 = np.random.rand(n) * 0.01  

# Solve the SIS ODE system
solution = solve_ivp(
    sis_dynamics, [0, t_max], x0, t_eval=np.linspace(0, t_max, 500), method='RK45'
)

# Extract solution: time series data (shape: time_steps x num_nodes)
sis_time_series = solution.y.T    
print(f"Generated SIS time series data shape: {sis_time_series.shape}")

# Prepare sliding window input-output pairs for RNN training
def prepare_rnn_data(data, window_size=10):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Input: window of 'window_size' steps
        Y.append(data[i+window_size])    # Output: next time step
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# Generate training data with window size 10
window_size = 10
X, Y = prepare_rnn_data(sis_time_series, window_size)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# -------------------------------
# Define GAT (Graph Attention Network) Components
# -------------------------------

# Single Graph Attention Layer
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads  # Number of attention heads
        self.out_features = out_features  # Features per head
        
        # Linear layer to project input features
        self.linear = torch.nn.Linear(in_features, out_features * num_heads, bias=False)
        
        # Attention parameters for source and destination nodes
        self.a_src = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_dst = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        
        # Xavier initialization
        torch.nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        
        self.leakyrelu = torch.nn.LeakyReLU(0.2)  # Activation function

    def forward(self, x, adj):
        batch_size = x.size(0)
        N = x.size(1)  # Number of nodes
        
        # Linear transformation
        h = self.linear(x)  # [batch_size, N, num_heads * out_features]
        h = h.view(batch_size, N, self.num_heads, self.out_features)
        
        # Attention coefficient calculation
        a_input_src = torch.einsum('bnhi,hio->bnh', h, self.a_src.permute(1, 2, 0))  
        a_input_dst = torch.einsum('bnhi,hio->bnh', h, self.a_dst.permute(1, 2, 0))  
        
        e = self.leakyrelu(a_input_src.unsqueeze(2) + a_input_dst.unsqueeze(1))  # Compatibility function
        
        # Mask non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # Expand adjacency for batch processing
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=2)  # Normalize
        
        # Apply attention weights to neighbor features
        h_prime = torch.einsum('bnmh,bmhi->bnhi', attention, h)
        
        return h_prime.reshape(batch_size, N, -1)  # Flatten heads into one vector

# Full GAT model with two attention layers
class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super(GAT, self).__init__()
        self.attention_layer1 = GraphAttentionLayer(in_features, hidden_features, num_heads)
        self.attention_layer2 = GraphAttentionLayer(hidden_features * num_heads, out_features, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x, adj):
        x = self.attention_layer1(x, adj)  # First attention layer
        x = self.activation(x)  # Activation
        x = self.attention_layer2(x, adj)  # Second attention layer
        return x

# Combined RNN + GAT model
class RNN_GAT_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads):
        super(RNN_GAT_Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN
        self.gat = GAT(hidden_size, gat_hidden, output_size, num_heads)  # (Optional) GAT (not used in final prediction)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = torch.nn.Linear(hidden_size, output_size)  # Final fully connected layer

    def forward(self, x, adj):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial hidden state
        rnn_out, _ = self.rnn(x, h0)  # Pass through RNN
        rnn_out = rnn_out[:, -1, :]  # Use output at last time step
        output = self.fc(rnn_out)  # Final output
        return output

# -------------------------------
# Training Setup
# -------------------------------

# Model parameters
input_size = n  # Number of nodes
hidden_size = 64  # Hidden size of RNN
output_size = n  # Predicting all nodes
num_layers = 2  # RNN layers
gat_hidden = 32  # Hidden units in GAT
num_heads = 4  # Number of attention heads
epochs = 2000  # Number of training epochs
lr = 0.01  # Learning rate
train_ratio = 0.5  # Ratio of training data

# Convert adjacency matrix to tensor
adj = torch.tensor(A, dtype=torch.float32)

# Split data into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Initialize model, loss, and optimizer
model = RNN_GAT_Model(input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads)
criterion = torch.nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

# -------------------------------
# Training Loop
# -------------------------------
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    Y_pred_train = model(X_train, adj)  # Forward pass
    loss = criterion(Y_pred_train, Y_train)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters
    losses.append(loss.item())  # Save loss

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")  # Print progress

# -------------------------------
# Testing Phase
# -------------------------------
with torch.no_grad():
    Y_pred_test = model(X_test, adj)  # Predict on test set
    test_loss = criterion(Y_pred_test, Y_test).item()  # Compute test loss
print(f"Test Loss: {test_loss}")

# Plot training loss over time
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# -------------------------------
# Plotting Predictions
# -------------------------------

# Compute average infection probabilities across all nodes
train_true_avg = Y_train.mean(dim=1).detach().numpy()
train_pred_avg = Y_pred_train.mean(dim=1).detach().numpy()
true_avg = Y_test.mean(dim=1).detach().numpy()
pred_avg = Y_pred_test.mean(dim=1).detach().numpy()

print(true_avg)
print(pred_avg)

# Plot average infection probabilities
plt.figure(figsize=(10, 5))
plt.plot(train_true_avg, label='Train True', marker='o', linestyle='None', markersize=4, markerfacecolor='none', markeredgecolor='yellow')
plt.plot(train_pred_avg, label='Train Predicted', marker='o', linestyle='None', markersize=1, markerfacecolor='none', markeredgecolor='red')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(true_avg)), true_avg, label='Test True Average')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(pred_avg)), pred_avg, label='Test Predicted Average', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Average Infection Probability')
plt.title('True vs Predicted Average Infection Probabilities (RNN-GAT Model)')
plt.legend()
plt.show()

# -------------------------------
# Scatter Plot: True vs Predicted Infection Probability
# -------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(true_avg, pred_avg, alpha=0.6, label='Predicted vs True')

# Calculate correlation coefficient
r_value, _ = pearsonr(true_avg, pred_avg)

# Get axis limits
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Set text location
x_text = x_min + 0.1 * (x_max - x_min)
y_text = y_min + 0.1 * (y_max - y_min)

# Show Pearson r value
plt.text(x_text, y_text, f'$r = {r_value:.4f}$', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Labels and formatting
plt.xlabel('True Infection Probability', fontsize=23, fontweight='bold')
plt.ylabel('Predicted Infection Probability', fontsize=23, fontweight='bold')
plt.title('Complex Network', fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

G = nx.erdos_renyi_graph(n, p)
A = nx.adjacency_matrix(G).toarray()
print(f"Number of nodes: {n}")

# SIS Model Parameters
beta = 0.012  # Infection rate
gamma = 0.03  # Recovery rate
t_max = 350  # Simulation time

# Define SIS model differential equations
def sis_dynamics(t, x):
    dxdt = np.zeros_like(x)
    for i in range(len(x)):
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)
        recovery_term = -gamma * x[i]
        dxdt[i] = infection_sum + recovery_term
    return dxdt

# Initial infection probabilities
x0 = np.random.rand(n) * 0.01  

# Solve the SIS model using ODE solver
solution = solve_ivp(
    sis_dynamics, [0, t_max], x0, t_eval=np.linspace(0, t_max, 500), method='RK45'
)

# Convert solution to time series data
sis_time_series = solution.y.T  # Shape: (time_steps, num_nodes)    
print(f"Generated SIS time series data shape: {sis_time_series.shape}")

def prepare_rnn_data(data, window_size=10):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# Prepare input-output pairs for training
window_size = 10
X, Y = prepare_rnn_data(sis_time_series, window_size)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# GAT Model Definition
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_src = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_dst = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        torch.nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        batch_size = x.size(0)
        N = x.size(1)  # Number of nodes
        
        # Linear transformation
        h = self.linear(x)  # [batch_size, N, num_heads * out_features]
        h = h.view(batch_size, N, self.num_heads, self.out_features)  # [batch_size, N, num_heads, out_features]
        
        # Compute attention coefficients
        a_input_src = torch.einsum('bnhi,hio->bnh', h, self.a_src.permute(1, 2, 0))  # [batch_size, N, num_heads]
        a_input_dst = torch.einsum('bnhi,hio->bnh', h, self.a_dst.permute(1, 2, 0))  # [batch_size, N, num_heads]
        
        e = self.leakyrelu(a_input_src.unsqueeze(2) + a_input_dst.unsqueeze(1))  # [batch_size, N, N, num_heads]
        
        # Mask out non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = torch.softmax(attention, dim=2)  # softmax over neighbors
        
        # Apply attention
        h_prime = torch.einsum('bnmh,bmhi->bnhi', attention, h)
        return h_prime.reshape(batch_size, N, -1)  # Flatten heads

class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super(GAT, self).__init__()
        self.attention_layer1 = GraphAttentionLayer(in_features, hidden_features, num_heads)
        self.attention_layer2 = GraphAttentionLayer(hidden_features * num_heads, out_features, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x, adj):
        x = self.attention_layer1(x, adj)
        x = self.activation(x)
        x = self.attention_layer2(x, adj)
        return x

# Combined RNN-GAT Model
class RNN_GAT_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads):
        super(RNN_GAT_Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.gat = GAT(hidden_size, gat_hidden, output_size, num_heads)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = torch.nn.Linear(hidden_size, output_size)  # Added final linear layer

    def forward(self, x, adj):
        # RNN processing
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        rnn_out, _ = self.rnn(x, h0)  # [batch_size, seq_len, hidden_size]
        rnn_out = rnn_out[:, -1, :]  # Take the last time step output [batch_size, hidden_size]
        
        # Pass through final linear layer
        output = self.fc(rnn_out)  # [batch_size, output_size]
        
        return output

# Model parameters
input_size = n
hidden_size = 64
output_size = n
num_layers = 2
gat_hidden = 32
num_heads = 4
epochs = 2000
lr = 0.01
train_ratio = 0.5

# Prepare adjacency matrix for GAT
adj = torch.tensor(A, dtype=torch.float32)

# Split into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]


# Initialize model
model = RNN_GAT_Model(input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred_train = model(X_train, adj)
    loss = criterion(Y_pred_train, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Testing phase
with torch.no_grad():
    Y_pred_test = model(X_test, adj)
    test_loss = criterion(Y_pred_test, Y_test).item()
print(f"Test Loss: {test_loss}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()



# Compute mean across all node features at each time step
train_true_avg = Y_train.mean(dim=1).detach().numpy()
train_pred_avg = Y_pred_train.mean(dim=1).detach().numpy()
true_avg = Y_test.mean(dim=1).detach().numpy()
pred_avg = Y_pred_test.mean(dim=1).detach().numpy()

print(true_avg)
print(pred_avg)

# Plot average infection probabilities
plt.figure(figsize=(10, 5))
plt.plot(train_true_avg, label='Train True', marker='o', linestyle='None', markersize=4, markerfacecolor='none', markeredgecolor='yellow')
plt.plot(train_pred_avg, label='Train Predicted', marker='o', linestyle='None', markersize=1, markerfacecolor='none', markeredgecolor='red')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(true_avg)), true_avg, label='Test True Average')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(pred_avg)), pred_avg, label='Test Predicted Average', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Average Infection Probability')
plt.title('True vs Predicted Average Infection Probabilities (RNN-GAT Model)')
plt.legend()
plt.show()





# Scatter plot of true vs predicted averages
plt.figure(figsize=(6, 6))
plt.scatter(true_avg, pred_avg, alpha=0.6, label='Predicted vs True')

# Compute correlation coefficient
r_value, _ = pearsonr(true_avg, pred_avg)

# Get axis limits
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Position text at 10% from bottom-left corner
x_text = x_min + 0.1 * (x_max - x_min)
y_text = y_min + 0.1 * (y_max - y_min)

# Add correlation coefficient
plt.text(x_text, y_text, f'$r = {r_value:.4f}$', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


# Labels and title
plt.xlabel('True Infection Probability', fontsize=23, fontweight='bold')
plt.ylabel('Predicted Infection Probability', fontsize=23, fontweight='bold')
plt.title('Complex Network', fontsize=24, fontweight='bold')

plt.xticks(fontsize=18, fontweight='bold')  # or any size you prefer
plt.yticks(fontsize=18, fontweight='bold')  # or any size you prefer

plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()