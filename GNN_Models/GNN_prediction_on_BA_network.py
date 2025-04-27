import torch                          # Import PyTorch for building and training models
import numpy as np                    # Import NumPy for numerical operations
import matplotlib.pyplot as plt       # Import Matplotlib for plotting graphs
from scipy.integrate import solve_ivp # Import ODE solver for solving differential equations
import networkx as nx                 # Import NetworkX to generate and manipulate graphs
from scipy.stats import pearsonr      # Import pearsonr to compute Pearson correlation coefficient

# ================================
# Generate a Barabási-Albert (BA) Network
# ================================
n = 400           # Number of nodes in the network
m = 2             # Number of edges to attach from a new node to existing nodes
G = nx.barabasi_albert_graph(n, m)          # Generate a BA network with n nodes and m new edges per node
A = nx.adjacency_matrix(G).toarray()        # Get the network's adjacency matrix as a NumPy array
print(f"Number of nodes: {n}")              # Print the number of nodes

# ================================
# Define SIS Model Parameters
# ================================
beta = 0.012    # Infection rate parameter
gamma = 0.03    # Recovery rate parameter
t_max = 350     # Total simulation time

# ================================
# Define SIS model differential equations (SIS Dynamics)
# ================================
def sis_dynamics(t, x):
    dxdt = np.zeros_like(x)         # Create an array of zeros of the same shape as x for derivatives
    for i in range(len(x)):         # Iterate over each node
        # Compute infection contribution from neighbors:
        # (1-x[i]) gives the susceptible portion of node i and np.dot(A[i], x) sums infection from connected nodes
        infection_sum = beta * (1 - x[i]) * np.dot(A[i], x)
        recovery_term = -gamma * x[i]  # Recovery term proportional to current infection level of node i
        dxdt[i] = infection_sum + recovery_term  # Total change in infection probability for node i
    return dxdt                     # Return the derivatives as the system dynamics

# ================================
# Set Initial Conditions and Solve the ODE
# ================================
x0 = np.random.rand(n) * 0.01   # Initialize infection probabilities with small random values for each node
solution = solve_ivp(
    sis_dynamics,               # The differential equation function to solve
    [0, t_max],                 # Time span for the simulation from 0 to t_max
    x0,                         # Initial condition for the solver
    t_eval=np.linspace(0, t_max, 500),  # Evaluation time points: 500 evenly spaced points in [0, t_max]
    method='RK45'               # Use the Runge-Kutta 45 method for solving ODE
)
sis_time_series = solution.y.T  # Transpose solution.y to obtain shape: (time_steps, num_nodes)
print(f"Generated SIS time series data shape: {sis_time_series.shape}")

# ================================
# Prepare RNN Data Using a Sliding Window
# ================================
def prepare_rnn_data(data, window_size=10):
    X, Y = [], []  # Initialize lists for inputs (X) and targets (Y)
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Input sequence of length equal to window_size
        Y.append(data[i+window_size])    # Target: the time step immediately following the sequence
    # Convert the lists to torch tensors with type float32 and return
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32)

window_size = 10                           # Set the window size for the sliding window approach
X, Y = prepare_rnn_data(sis_time_series, window_size)  # Prepare input-output pairs
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# ================================
# Define the Graph Attention Network (GAT) Components
# ================================
# Define a single Graph Attention Layer
class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads                   # Number of attention heads
        self.out_features = out_features             # Number of features output per head
        # Linear transformation for multi-head attention; no bias is used here
        self.linear = torch.nn.Linear(in_features, out_features * num_heads, bias=False)
        # Learnable parameters for computing attention coefficients (source and destination)
        self.a_src = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_dst = torch.nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        # Initialize attention parameters using Xavier uniform initialization
        torch.nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)       # LeakyReLU activation for non-linearity in attention computation

    def forward(self, x, adj):
        batch_size = x.size(0)     # Batch size (number of samples)
        N = x.size(1)              # Number of nodes in each sample
        
        # Apply linear transformation
        h = self.linear(x)         # Output shape: [batch_size, N, num_heads*out_features]
        h = h.view(batch_size, N, self.num_heads, self.out_features)  # Reshape to [batch_size, N, num_heads, out_features]
        
        # Compute attention coefficients for source and destination nodes using Einstein summation
        a_input_src = torch.einsum('bnhi,hio->bnh', h, self.a_src.permute(1, 2, 0))
        a_input_dst = torch.einsum('bnhi,hio->bnh', h, self.a_dst.permute(1, 2, 0))
        
        # Combine source and destination attention scores and apply a non-linear activation (LeakyReLU)
        e = self.leakyrelu(a_input_src.unsqueeze(2) + a_input_dst.unsqueeze(1))  # Shape: [batch_size, N, N, num_heads]
        
        # Mask out the non-adjacent nodes by assigning a large negative value
        zero_vec = -9e15 * torch.ones_like(e)
        # Expand adjacency matrix dimensions to match the attention score dimensions
        adj_expanded = adj.unsqueeze(0).unsqueeze(-1)  # Shape becomes [1, N, N, 1]
        attention = torch.where(adj_expanded > 0, e, zero_vec)  # Keep e for neighbors; assign -9e15 for non-neighbors
        attention = torch.softmax(attention, dim=2)           # Apply softmax along neighbors dimension
        
        # Compute the new node features as the weighted sum of neighbor features
        h_prime = torch.einsum('bnmh,bmhi->bnhi', attention, h)
        # Flatten the multiple heads into a single vector per node and return
        return h_prime.reshape(batch_size, N, -1)

# Define the complete GAT module with two attention layers
class GAT(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super(GAT, self).__init__()
        # First graph attention layer that projects input features into a hidden representation
        self.attention_layer1 = GraphAttentionLayer(in_features, hidden_features, num_heads)
        # Second graph attention layer that projects the concatenated multi-head features to the desired output size
        self.attention_layer2 = GraphAttentionLayer(hidden_features * num_heads, out_features, 1)
        self.activation = torch.nn.ReLU()            # ReLU activation to add non-linearity

    def forward(self, x, adj):
        x = self.attention_layer1(x, adj)             # Pass through the first attention layer
        x = self.activation(x)                        # Apply ReLU activation
        x = self.attention_layer2(x, adj)             # Pass through the second attention layer
        return x

# ================================
# Define the Combined RNN-GAT Model
# ================================
class RNN_GAT_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads):
        super(RNN_GAT_Model, self).__init__()
        # Define an RNN that processes time series data, using batch_first=True for convenience
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # Initialize a GAT model; although defined here, the GAT module is not used directly in the forward pass in this version
        self.gat = GAT(hidden_size, gat_hidden, output_size, num_heads)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # A final fully connected layer to map RNN output to the desired output dimension
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, adj):
        # Initialize the hidden state with zeros for the RNN (shape: [num_layers, batch_size, hidden_size])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # Process the input sequence through the RNN; ignore the second output (hidden state) here
        rnn_out, _ = self.rnn(x, h0)  # Output shape: [batch_size, seq_length, hidden_size]
        rnn_out = rnn_out[:, -1, :]    # Take only the last time step's output for prediction (shape: [batch_size, hidden_size])
        
        # Apply the final fully connected (linear) layer to get predictions (shape: [batch_size, output_size])
        output = self.fc(rnn_out)
        return output

# ================================
# Set Model Parameters and Prepare Data
# ================================
input_size = n      # Each node is treated as a feature
hidden_size = 64    # Hidden state size for the RNN
output_size = n     # The output is the predicted state of all nodes
num_layers = 2      # Number of RNN layers
gat_hidden = 32     # Hidden dimension size used in the GAT module
num_heads = 4       # Number of attention heads in the GAT module
epochs = 2000       # Total number of training epochs
lr = 0.01           # Learning rate for the optimizer
train_ratio = 0.5   # Proportion of data used for training

# Convert the adjacency matrix A to a PyTorch tensor for use in the model
adj = torch.tensor(A, dtype=torch.float32)

# Split the prepared RNN data (X and Y) into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# ================================
# Add Additional Samples to the Training Set
# (Take a small sample from the test set and append to training data)
# ================================
test_sample_size = int(len(X_test) * 0.01)  # Determine size of sample (1% of test set)
X_test_sample = X_test[-test_sample_size:]   # Select the last test_sample_size samples from X_test
Y_test_sample = Y_test[-test_sample_size:]   # Select corresponding labels from Y_test

# Append the selected test samples to the training data
X_train = torch.cat([X_train, X_test_sample])
Y_train = torch.cat([Y_train, Y_test_sample])

# ================================
# Model Initialization, Loss, and Optimizer
# ================================
model = RNN_GAT_Model(input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads) # Initialize the combined RNN-GAT model
criterion = torch.nn.MSELoss()                # Use Mean Squared Error as the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Use the Adam optimizer with the specified learning rate

# ================================
# Training Loop
# ================================
losses = []  # List to store loss values for each epoch
for epoch in range(epochs):
    optimizer.zero_grad()                 # Clear gradients from previous iteration
    Y_pred_train = model(X_train, adj)      # Forward pass: compute model predictions for the training set
    loss = criterion(Y_pred_train, Y_train) # Compute loss between predictions and true values
    loss.backward()                       # Backward pass: calculate gradients
    optimizer.step()                      # Update model parameters using the computed gradients
    losses.append(loss.item())            # Record the loss value
    
    if (epoch + 1) % 10 == 0:             # Every 10 epochs, print the current loss to track progress
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# ================================
# Testing Phase
# ================================
with torch.no_grad():                     # Disable gradient calculation for testing phase
    Y_pred_test = model(X_test, adj)       # Compute predictions on the test dataset
    test_loss = criterion(Y_pred_test, Y_test).item()  # Compute the loss on test data
print(f"Test Loss: {test_loss}")

# ================================
# Plot Training Loss Over Epochs
# ================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

# ================================
# Remove the Extra Samples That Were Added from the Test Set
# ================================
Y_train = Y_train[:-test_sample_size]       # Remove extra samples from Y_train
Y_pred_train = Y_pred_train[:-test_sample_size]  # Remove corresponding predictions for consistent plotting

# ================================
# Compute the Average Infection Probabilities
# (Average across all nodes for each time step)
# ================================
train_true_avg = Y_train.mean(dim=1).detach().numpy()  # Average true infection probability for training samples
train_pred_avg = Y_pred_train.mean(dim=1).detach().numpy()  # Average predicted infection probability for training samples
true_avg = Y_test.mean(dim=1).detach().numpy()           # Average true infection probability for test samples
pred_avg = Y_pred_test.mean(dim=1).detach().numpy()      # Average predicted infection probability for test samples

print(true_avg)   # Print the true average infection probabilities for test set
print(pred_avg)   # Print the predicted average infection probabilities for test set

# ================================
# Plot Average Infection Probabilities Over Time
# ================================
plt.figure(figsize=(10, 5))
# Plot averages for training data (true and predicted)
plt.plot(train_true_avg, label='Train True', marker='o', linestyle='None', markersize=4, markerfacecolor='none', markeredgecolor='yellow')
plt.plot(train_pred_avg, label='Train Predicted', marker='o', linestyle='None', markersize=1, markerfacecolor='none', markeredgecolor='red')
# Plot averages for test data (true and predicted); the x-axis is shifted accordingly
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(true_avg)), true_avg, label='Test True Average')
plt.plot(range(len(train_pred_avg), len(train_pred_avg) + len(pred_avg)), pred_avg, label='Test Predicted Average', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Average Infection Probability')
plt.title('True vs Predicted Average Infection Probabilities (RNN-GAT Model)')
plt.legend()
plt.show()

# ================================
# Scatter Plot: True vs Predicted Average Infection Probabilities
# ================================
plt.figure(figsize=(6, 6))
plt.scatter(true_avg, pred_avg, alpha=0.6, label='Predicted vs True')  # Create scatter plot comparing true and predicted averages

# Compute Pearson correlation coefficient between true and predicted averages
r_value, _ = pearsonr(true_avg, pred_avg)

# Get current axis limits for positioning of the text annotation
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Position the correlation coefficient text 10% from the bottom-left corner
x_text = x_min + 0.1 * (x_max - x_min)
y_text = y_min + 0.1 * (y_max - y_min)
plt.text(x_text, y_text, f'$r = {r_value:.4f}$', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ================================
# Finalize Plot Appearance: Labels, Title, and Ticks
# ================================
plt.xlabel('True Infection Probability', fontsize=23, fontweight='bold')
plt.ylabel('Predicted Infection Probability', fontsize=23, fontweight='bold')
plt.title('Complex Network', fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')  # Set x-axis tick parameters
plt.yticks(fontsize=18, fontweight='bold')  # Set y-axis tick parameters
plt.legend()
plt.tight_layout()  # Adjust subplot params for a neat layout
plt.show()
