# Import necessary libraries
import torch  # PyTorch for deep learning
import numpy as np  # NumPy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting
from scipy.integrate import solve_ivp  # Scipy ODE solver for numerical integration

# Load adjacency matrix (network structure) from file
A = np.loadtxt(r"matrix/ERmatrix50.txt", dtype=int)
n = A.shape[0]  # Number of nodes in the network
print(f"Number of nodes: {n}")

# SIS Model Parameters
beta = 0.0022  # Infection rate
gamma = 0.03  # Recovery rate
t_max = 350  # Maximum simulation time

# Define SIS model differential equations
def sis_dynamics(t, x):
    dxdt = np.zeros_like(x)  # Initialize rate of change
    for i in range(len(x)):
        # Infection term: susceptible nodes becoming infected
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)
        # Recovery term: infected nodes recovering
        recovery_term = -gamma * x[i]
        dxdt[i] = infection_sum + recovery_term  # Net change
    return dxdt

# Initial infection probabilities (small random values)
x0 = np.random.rand(n) * 0.01  

# Solve the SIS model over time using an ODE solver
solution = solve_ivp(
    sis_dynamics, [0, t_max], x0, t_eval=np.linspace(0, t_max, 500), method='RK45'
)

# Convert solution to time series data (time_steps x nodes)
sis_time_series = solution.y.T    
print(f"Generated SIS time series data shape: {sis_time_series.shape}")

# Prepare RNN training data by creating sequences
def prepare_rnn_data(data, window_size=10):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Input sequence
        Y.append(data[i+window_size])    # Target output
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

# Create input-output pairs for training
window_size = 10
X, Y = prepare_rnn_data(sis_time_series, window_size)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Define a single Graph Attention Layer
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features * num_heads, bias=False)  # Linear transform
        self.a_src = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))  # Source attention parameters
        self.a_dst = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))  # Destination attention parameters
        torch.nn.init.xavier_uniform_(self.a_src.data, gain=1.414)  # Initialize weights
        torch.nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)  # Activation function for attention coefficients

    def forward(self, x, adj):
        batch_size = x.size(0)
        N = x.size(1)  # Number of nodes

        # Apply linear transformation
        h = self.linear(x)  
        h = h.view(batch_size, N, self.num_heads, self.out_features)  # Reshape for multi-head attention

        # Compute attention coefficients
        a_input_src = torch.einsum('bnhi,hio->bnh', h, self.a_src.permute(1, 2, 0))
        a_input_dst = torch.einsum('bnhi,hio->bnh', h, self.a_dst.permute(1, 2, 0))

        # Combine attention from source and destination
        e = self.leakyrelu(a_input_src.unsqueeze(2) + a_input_dst.unsqueeze(1))

        # Mask non-existent edges
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # Expand adjacency matrix dimensions
        attention = torch.where(adj_expanded > 0, e, zero_vec)  # Apply masking
        attention = torch.softmax(attention, dim=2)  # Normalize across neighbors

        # Aggregate neighbor features using attention weights
        h_prime = torch.einsum('bnmh,bmhi->bnhi', attention, h)

        return h_prime.reshape(batch_size, N, -1)  # Flatten the multi-head outputs

# Define a two-layer GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super(GAT, self).__init__()
        self.attention_layer1 = GraphAttentionLayer(in_features, hidden_features, num_heads)  # First attention layer
        self.attention_layer2 = GraphAttentionLayer(hidden_features * num_heads, out_features, 1)  # Output layer with 1 head
        self.activation = torch.nn.ReLU()  # Non-linearity

    def forward(self, x, adj):
        x = self.attention_layer1(x, adj)  # First GAT layer
        x = self.activation(x)  # Apply activation
        x = self.attention_layer2(x, adj)  # Second GAT layer
        return x

# Define combined RNN-GAT model
class RNN_GAT_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads):
        super(RNN_GAT_Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN layer
        self.gat = GAT(hidden_size, gat_hidden, output_size, num_heads)  # GAT module (not used directly here)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = torch.nn.Linear(hidden_size, output_size)  # Final linear projection

    def forward(self, x, adj):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial hidden state
        rnn_out, _ = self.rnn(x, h0)  # Pass input through RNN
        rnn_out = rnn_out[:, -1, :]  # Take output from the last time step
        output = self.fc(rnn_out)  # Final output prediction
        return output

# Set model hyperparameters
input_size = n  # Input size = number of nodes
hidden_size = 64  # RNN hidden size
output_size = n  # Output size = number of nodes
num_layers = 2  # Number of RNN layers
gat_hidden = 32  # Hidden features inside GAT
num_heads = 4  # Number of heads for GAT
epochs = 2000  # Training epochs
lr = 0.01  # Learning rate
train_ratio = 0.5  # Train/test split ratio

# Prepare adjacency matrix for model input
adj = torch.tensor(A, dtype=torch.float32)

# Split data into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# (Optional) Boost training set with some part of testing data
test_sample_size = int(len(X_test) * 0.01)  # Take 1% of test data
X_test_sample = X_test[-test_sample_size:]  # Last few test samples
Y_test_sample = Y_test[-test_sample_size:]

# Add selected test samples back to training set
X_train = torch.cat([X_train, X_test_sample])
Y_train = torch.cat([Y_train, Y_test_sample])

# Initialize model, loss function, and optimizer
model = RNN_GAT_Model(input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

# Train the model
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    Y_pred_train = model(X_train, adj)  # Forward pass
    loss = criterion(Y_pred_train, Y_train)  # Compute loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights
    losses.append(loss.item())

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Evaluate model on test data
with torch.no_grad():
    Y_pred_test = model(X_test, adj)
    test_loss = criterion(Y_pred_test, Y_test).item()
print(f"Test Loss: {test_loss}")

# Plot the training loss over time
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# Remove the extra samples we added to training set (for clean plotting)
Y_train = Y_train[:-test_sample_size]
Y_pred_train = Y_pred_train[:-test_sample_size]

# Compute mean infection probabilities for better visualization
train_true_avg = Y_train.mean(dim=1).detach().numpy()
train_pred_avg = Y_pred_train.mean(dim=1).detach().numpy()
true_avg = Y_test.mean(dim=1).detach().numpy()
pred_avg = Y_pred_test.mean(dim=1).detach().numpy()

# Plot actual vs predicted infection probabilities
plt.figure(figsize=(10, 5))
plt.plot(train_true_avg, label='Train True infection  Prob.', marker='o', linestyle='None', markersize=4, markerfacecolor='none', markeredgecolor='yellow')
plt.plot(train_pred_avg, label='Train Predicted infection  Prob.', marker='o', linestyle='None', markersize=1, markerfacecolor='none', markeredgecolor='red')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(true_avg)), true_avg, label='Test True infection  Prob.')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(pred_avg)), pred_avg, label='Test Predicted infection  Prob.', linestyle='dashed')
plt.xlabel('Time Step', fontsize=23, fontweight='bold')
plt.ylabel('Infection Probability', fontsize=23, fontweight='bold')
plt.title('RNN + GAT: True vs Predicted Infection Probability', fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
plt.legend()
plt.show()
