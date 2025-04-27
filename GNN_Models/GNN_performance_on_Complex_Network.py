# Import necessary libraries
import numpy as np  # For numerical operations like arrays, matrix operations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import networkx as nx  # For creating and manipulating graphs
import torch  # For working with PyTorch (neural networks, tensors)
import torch.nn as nn  # For defining neural network layers
import torch.nn.functional as F  # For using various PyTorch functions like activation functions

# Define ComplexContagion class for modeling complex contagion dynamics
class ComplexContagion:
    def __init__(self, temperature=2.0, recovery=0.02):  # Initialize parameters like temperature and recovery rate
        self.temperature = temperature  # Set temperature for the infection function
        self.recovery = recovery  # Set recovery rate
    
    def activation(self, l):
        """Planck-like infection function with proper non-monotonic behavior"""
        l = np.array(l, dtype=float)  # Convert input list to a numpy array (for vectorized operations)
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for invalid or division-by-zero operations
            exp_term = np.exp(l/self.temperature)  # Compute the exponential term in the Planck-like equation
            result = (l**2) / (exp_term - 1)  # Calculate the result according to the Planck-like formula
            result[exp_term == 1] = 0  # Handle division by zero (if exp_term equals 1)
            result[np.isnan(result)] = 0  # Replace NaN values with zeros
            return result / self._normalization()  # Normalize the result using the normalization factor
    
    def _normalization(self):
        """Normalization tuned for peak around ℓ=8-10"""
        l_max = self.temperature * 1.5  # Adjust the peak location of the distribution
        return (l_max**2) / (np.exp(l_max/self.temperature) - 1)  # Calculate the normalization factor based on peak location
    
    def deactivation(self, l):
        """Constant recovery probability"""
        return np.full_like(l, self.recovery, dtype=float)  # Return a constant recovery probability for all nodes

# Define the Graph Attention Layer (GAT)
class GATLayer(nn.Module):
    """Properly implemented Graph Attention Layer"""
    def __init__(self, in_features, out_features):
        super().__init__()  # Initialize parent class nn.Module
        self.W = nn.Linear(in_features, out_features)  # Linear transformation for input features
        self.a = nn.Linear(2 * out_features, 1)  # Linear transformation for attention coefficients
        self.leakyrelu = nn.LeakyReLU(0.2)  # Leaky ReLU activation for attention scores
    
    def forward(self, x, edge_index):
        x = self.W(x)  # Apply linear transformation to node features
        row, col = edge_index  # Unpack the edge indices (pairs of connected nodes)
        
        # Compute attention coefficients for each edge
        x_row = x[row]  # Features of the source nodes (row)
        x_col = x[col]  # Features of the target nodes (col)
        alpha = self.leakyrelu(self.a(torch.cat([x_row, x_col], dim=1)))  # Compute attention score using concatenation of source and target features
        alpha = F.softmax(alpha, dim=0)  # Normalize the attention coefficients using softmax
        
        # Aggregate features using attention scores
        out = torch.zeros_like(x)  # Initialize an empty tensor for the output
        out = out.index_add_(0, row, alpha * x_col)  # Add the weighted target features to the output
        return out  # Return the aggregated features

# Define the Stochastic Epidemics GNN model
class StochasticEpidemicsGNN(nn.Module):
    def __init__(self, num_states=2, hidden_size=32):
        super().__init__()  # Initialize parent class nn.Module
        self.num_states = num_states  # Set the number of possible states (e.g., infected or susceptible)
        self.input_proj = nn.Linear(1, hidden_size)  # Linear projection for the input feature (node state)
        self.gnn = GATLayer(hidden_size, hidden_size)  # Graph Attention Layer
        self.mlp = nn.Sequential(  # Multi-layer Perceptron (MLP) for final classification
            nn.Linear(hidden_size, hidden_size),  # Linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hidden_size, num_states)  # Output layer for predicting states
        )
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)  # Apply linear transformation to input node features
        h = self.gnn(x, edge_index)  # Apply the Graph Attention Layer for message passing
        return F.softmax(self.mlp(h), dim=-1)  # Apply MLP and softmax for state classification

# Generate a Barabási-Albert (BA) network (scale-free network)
num_nodes = 1000  # Number of nodes in the network
avg_degree = 4  # Average degree (edges per node)
G = nx.barabasi_albert_graph(num_nodes, avg_degree // 2)  # Generate BA graph with specified average degree
edge_index = torch.tensor(list(G.edges)).t().contiguous()  # Convert edge list to tensor (format for PyTorch)

# Initialize the dynamics of the contagion process
dynamics = ComplexContagion(temperature=2.0, recovery=0.02)  # Set temperature and recovery rate for the complex contagion

# Function to simulate the dynamics of the epidemic on the graph
def simulate_dynamics(dynamics, G, steps=500):
    states = []  # List to store states at each step
    x = torch.zeros(num_nodes)  # Initialize node states (0 for susceptible)
    x[torch.randperm(num_nodes)[:10]] = 1  # Randomly infect 10 nodes initially
    
    for _ in range(steps):
        l = torch.zeros(num_nodes)  # Initialize a tensor to store infected neighbor counts
        for i, j in G.edges():  # Loop over all edges in the graph
            l[i] += x[j]  # Count infected neighbors for node i
            l[j] += x[i]  # Count infected neighbors for node j
            
        # Compute transition probabilities for infection and recovery
        with torch.no_grad():
            p_infect = torch.tensor(dynamics.activation(l.numpy()), dtype=torch.float32)  # Infection probabilities
            p_recover = torch.tensor(dynamics.deactivation(l.numpy()), dtype=torch.float32)  # Recovery probabilities
        
        # Sample the next state based on current probabilities
        y = x.clone()  # Create a copy of the current state
        infect_mask = (x == 0) & (torch.rand(num_nodes) < p_infect)  # Nodes that should get infected
        recover_mask = (x == 1) & (torch.rand(num_nodes) < p_recover)  # Nodes that should recover
        y[infect_mask] = 1  # Update state to infected for selected nodes
        y[recover_mask] = 0  # Update state to susceptible for recovered nodes
        
        states.append((x.clone(), y.clone(), l.clone()))  # Store the state at the current step
        x = y  # Update the state for the next step
        
    return states  # Return the simulated states

# Generate training data by simulating dynamics on the graph
data = simulate_dynamics(dynamics, G)

# Define the training process for the Stochastic Epidemics GNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select device (GPU if available)
model = StochasticEpidemicsGNN().to(device)  # Instantiate the model and move it to the selected device
edge_index = edge_index.to(device)  # Move edge_index to the same device
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Set up the Adam optimizer

# Train the model
for epoch in range(10):  # Train for 10 epochs (reduced for demonstration)
    model.train()  # Set model to training mode
    for x, y, _ in data[:100]:  # Use a subset of the data for training (first 100 samples)
        x = x.float().unsqueeze(1).to(device)  # Convert the input state to a float tensor and move to device
        y = y.long().to(device)  # Convert the target state to a long tensor and move to device
        
        optimizer.zero_grad()  # Zero the gradients
        pred = model(x, edge_index)  # Get model predictions
        loss = F.cross_entropy(pred, y)  # Compute the cross-entropy loss
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

# Evaluate the model
model.eval()  # Set model to evaluation mode
l_values = np.arange(0, 20)  # Range of infected neighbor values
gnn_probs = np.zeros((len(l_values), 2, 2))  # Array to store GNN probabilities
mle_counts = np.zeros((len(l_values), 2, 2))  # Array to store MLE counts

# Count transitions for Maximum Likelihood Estimation (MLE)
for x, y, l in data:
    x = x.numpy().astype(int)  # Convert states to integers
    y = y.numpy().astype(int)
    l = l.numpy().astype(int)
    
    for node in range(num_nodes):
        l_int = min(int(l[node]), len(l_values)-1)  # Ensure l is within bounds
        current_state = min(x[node], 1)  # Ensure current state is either 0 or 1
        next_state = min(y[node], 1)  # Ensure next state is either 0 or 1
        
        mle_counts[l_int, current_state, next_state] += 1  # Update MLE counts

# Calculate MLE probabilities from the counts
mle_probs = np.zeros_like(mle_counts)  # Initialize MLE probabilities
for l in range(len(l_values)):
    for s in range(2):
        total = mle_counts[l, s].sum()  # Get the total number of samples for the current state
        if total > 0:
            mle_probs[l, s] = mle_counts[l, s] / total  # Calculate probabilities

# Get GNN predictions
with torch.no_grad():  # Disable gradient computation during evaluation
    for l in l_values:
        test_x = torch.zeros(num_nodes, 1, device=device)  # Initialize test input (all susceptible)
        test_x[:l] = 1  # Set the first l nodes to infected
        pred = model(test_x, edge_index).mean(0).cpu().numpy()  # Get the mean prediction for each node
        gnn_probs[l] = pred  # Store the GNN probabilities

# Plot results
# Plotting parameters to match paper's style
plt.figure(figsize=(8, 5))  # Set the figure size for the plot
l_cont = np.linspace(0, 20, 100)  # Continuous range of infected neighbor values for plotting

# Ground truth infection and recovery probabilities
plt.plot(l_cont, dynamics.activation(l_cont), 'b-', linewidth=2, label='GT Infection (S→I)')
plt.axhline(dynamics.recovery, color='r', linestyle='-', linewidth=2, label='GT Recovery (I→S)')

# Plot GNN predictions
plt.plot(l_values, gnn_probs[:, 0, 1], 'b--', linewidth=2, label='GNN Infection (S→I)')
plt.plot(l_values, gnn_probs[:, 1, 0], 'r--', linewidth=2, label='GNN Recovery (I→S)')

# Plot MLE estimates with error bars
valid_l = np.where(mle_counts[:, 0].sum(1) > 10)[0]  # Filter valid values with sufficient samples
plt.errorbar(valid_l, mle_probs[valid_l, 0, 1], 
             yerr=np.sqrt(mle_probs[valid_l, 0, 1]*(1-mle_probs[valid_l, 0, 1])/mle_counts[valid_l, 0].sum(1)),
             fmt='bo', markersize=6, capsize=4, label='MLE Infection (S→I)')

valid_l = np.where(mle_counts[:, 1].sum(1) > 10)[0]  # Filter valid values with sufficient samples
plt.errorbar(valid_l, mle_probs[valid_l, 1, 0], 
             yerr=np.sqrt(mle_probs[valid_l, 1, 0]*(1-mle_probs[valid_l, 1, 0])/mle_counts[valid_l, 1].sum(1)),
             fmt='ro', markersize=6, capsize=4, label='MLE Recovery (I→S)')

# Formatting the plot to match the paper's style
plt.xlabel('Number of infected neighbors (ℓ)', fontsize=23, fontweight='bold')
plt.ylabel('Transition probability', fontsize=23, fontweight='bold')
plt.title('Complex Contagion Dynamics', fontsize=24, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')  # Adjust font size and weight for x-axis ticks
plt.yticks(fontsize=18, fontweight='bold')  # Adjust font size and weight for y-axis ticks
plt.legend(fontsize=15, framealpha=0.9)  # Add legend with adjusted font size and transparency
plt.xticks(np.arange(0, 21, 2))  # Set x-axis ticks with a step of 2
plt.xlim(0, 20)  # Set the x-axis limits
plt.ylim(0, 1)  # Set the y-axis limits
plt.tight_layout()  # Adjust layout to avoid clipping
plt.show()  # Display the plot
