# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx

# =========================
# 1. Implement Config Class
# =========================
class Config:
    def __init__(self, **kwargs):
        # Initialize the configuration class by setting attributes from kwargs
        self.__dict__.update(kwargs)

# ==============================
# 2. Implement Network Generator
# ==============================
def generate_barabasi_albert_network(num_nodes, m):
    """Generate a Barabási-Albert (scale-free) network using NetworkX."""
    G = nx.barabasi_albert_graph(n=num_nodes, m=m)  # Create BA graph
    edge_index = torch.tensor(list(G.edges)).t().contiguous()  # Convert edges to tensor form (for PyTorch GNNs)
    return G, edge_index  # Return both NetworkX graph and PyTorch edge list

# =========================
# 3. Implement SIS Dynamics
# =========================
class SISDynamics:
    def __init__(self, infection_rate, recovery_rate):
        # Initialize SIS dynamics parameters
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
    
    def simulate(self, network, num_steps):
        """Simulate SIS dynamics for a given network over num_steps time."""
        edge_index = torch.tensor(list(network.edges)).t().contiguous()  # Get edge index
        num_nodes = network.number_of_nodes()  # Total number of nodes
        
        # Initialize list of node states over time (0=S, 1=I)
        states = [torch.zeros(num_steps+1, dtype=torch.long) for _ in range(num_nodes)]
        for i in range(num_nodes):
            states[i][0] = torch.randint(0, 2, (1,))  # Random initial state (S or I)
        
        # Iterate through each time step
        for t in range(num_steps):
            for i in range(num_nodes):
                if states[i][t] == 0:  # If node is Susceptible
                    neighbors = edge_index[1][edge_index[0] == i]  # Find neighbors
                    infected_neighbors = sum(states[j][t] for j in neighbors)  # Count infected neighbors
                    # Infection probability: 1 - (1-γ)^ℓ
                    if torch.rand(1) < 1 - (1 - self.infection_rate)**infected_neighbors:
                        states[i][t+1] = 1  # Becomes infected
                    else:
                        states[i][t+1] = 0  # Remains susceptible
                else:  # If node is Infected
                    if torch.rand(1) < self.recovery_rate:
                        states[i][t+1] = 0  # Recovers to Susceptible
                    else:
                        states[i][t+1] = 1  # Remains infected
        
        # Stack the states over time into a tensor
        state_tensor = torch.stack([s[:-1] for s in states], dim=1)  # Current states
        next_state_tensor = torch.stack([s[1:] for s in states], dim=1)  # Next states
        return state_tensor, next_state_tensor  # Return tensors

# ==============================
# 4. Implement Dataset Wrapper
# ==============================
class DynamicsDataset:
    def __init__(self, states, network):
        # Initialize dataset
        self.states = states[0]  # Current states
        self.next_states = states[1]  # Next time-step states
        self.network = network  # Underlying network
        self.edge_index = torch.tensor(list(network.edges)).t().contiguous()  # Edge index
        self.num_nodes = network.number_of_nodes()  # Number of nodes
        self.num_steps = self.states.shape[0]  # Number of time steps
    
    def __len__(self):
        return self.num_steps  # Dataset length = number of time steps
    
    def __getitem__(self, idx):
        # Return (current state, next state, node attributes) at time idx
        return (self.states[idx], self.next_states[idx], torch.ones(self.num_nodes))

# =================================
# 5. Implement Stochastic GNN Model
# =================================
class StochasticEpidemicsGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Model hyperparameters
        self.num_states = config.num_states
        self.lag = config.lag
        
        # Define simple 2-layer GNN (here as Linear layers since edges aren't directly used)
        self.conv1 = nn.Linear(1, 16)  # First layer (input=1 feature)
        self.conv2 = nn.Linear(16, self.num_states)  # Output layer (output=number of states)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for probability outputs
        
    def forward(self, x, network_attr):
        # Forward pass through the GNN
        edge_index, edge_attr, node_attr = network_attr  # Unpack network attributes
        x = x.float().view(-1, 1)  # Ensure x has correct shape
        x = torch.relu(self.conv1(x))  # First layer + ReLU activation
        x = self.conv2(x)  # Second layer
        return self.softmax(x)  # Apply softmax to output probabilities

# ======================
# 6. Training Procedure
# ======================
def train_model(model, dataset, epochs=30, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()  # Loss function
    
    model.train()  # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0  # Accumulate loss over epoch
        for step in range(0, len(dataset), batch_size):
            batch = dataset[step:step+batch_size]  # Mini-batch
            x, y, _ = batch  # Unpack batch
            
            optimizer.zero_grad()  # Reset gradients
            pred = model(x, (dataset.edge_index, None, None))  # Forward pass
            loss = criterion(pred.view(-1, 2), y.view(-1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            
            total_loss += loss.item()  # Track loss
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")  # Print average loss

# =========================
# 7. Generate Figure 
# =========================
def generate_figure():
    # ----------------
    # Data Preparation
    # ----------------
    num_nodes = 1000  # Number of nodes
    avg_degree = 4  # Average degree (approximate)
    infection_rate = 0.2  # Infection rate γ
    recovery_rate = 0.1  # Recovery rate β
    num_steps = 5000  # Number of time steps
    
    network, edge_index = generate_barabasi_albert_network(num_nodes, avg_degree//2)  # Generate network
    dynamics = SISDynamics(infection_rate, recovery_rate)  # Initialize SIS dynamics
    states = dynamics.simulate(network, num_steps)  # Simulate dynamics
    dataset = DynamicsDataset(states, network)  # Wrap into a dataset
    
    # ----------------
    # Model Training
    # ----------------
    config = Config(num_states=2, lag=1)  # Model configuration
    model = StochasticEpidemicsGNN(config)  # Initialize model
    train_model(model, dataset)  # Train model on the dataset
    
    # ----------------
    # Compute Estimates
    # ----------------
    def ground_truth_SIS(l_values, beta=recovery_rate, gamma=infection_rate):
        """Calculate ground-truth transition probabilities based on SIS model."""
        infection_prob = 1 - (1 - gamma) ** l_values  # S->I probability
        recovery_prob = np.full_like(l_values, beta)  # I->S probability (constant)
        return infection_prob, recovery_prob
    
    def mle_estimates(dataset, max_l=10):
        """Estimate transition probabilities empirically from data."""
        counts_inf = np.zeros(max_l + 1)
        counts_rec = np.zeros(max_l + 1)
        total_inf = np.zeros(max_l + 1)
        total_rec = np.zeros(max_l + 1)
        
        for t in range(len(dataset)):
            x, y = dataset.states[t], dataset.next_states[t]
            for i in range(dataset.num_nodes):
                neighbors = dataset.edge_index[1][dataset.edge_index[0] == i]  # Get neighbors
                l = (x[neighbors.numpy()] == 1).sum()  # Count infected neighbors
                l = min(l, max_l)  # Cap at max_l
                
                if x[i] == 0 and y[i] == 1:  # S→I transition
                    counts_inf[l] += 1
                elif x[i] == 1 and y[i] == 0:  # I→S transition
                    counts_rec[l] += 1
                
                if x[i] == 0:
                    total_inf[l] += 1
                elif x[i] == 1:
                    total_rec[l] += 1
        
        # Calculate probabilities safely (avoid division by zero)
        infection_mle = np.divide(counts_inf, total_inf, out=np.zeros_like(counts_inf), where=total_inf != 0)
        recovery_mle = np.divide(counts_rec, total_rec, out=np.zeros_like(counts_rec), where=total_rec != 0)
        return infection_mle, recovery_mle, total_inf, total_rec
    
    l_values = np.arange(0, 11)  # Range of infected neighbors ℓ
    infection_mle, recovery_mle, total_inf, total_rec = mle_estimates(dataset)  # MLE estimates
    gt_inf, gt_rec = ground_truth_SIS(l_values)  # Ground-truth values
    
    # ----------------
    # GNN Predictions
    # ----------------
    infection_gnn, recovery_gnn = [], []
    for l in l_values:
        x = torch.zeros(1, 1)  # State: Susceptible
        neighbors = torch.ones(l, 1)  # l infected neighbors
        pred = model(x, (edge_index, None, neighbors))  # Model prediction
        infection_gnn.append(pred[0, 1].item())  # S->I probability
        
        x = torch.ones(1, 1)  # State: Infected
        pred = model(x, (edge_index, None, neighbors))
        recovery_gnn.append(pred[0, 0].item())  # I->S probability
    
    # ----------------
    # Plot Results
    # ----------------
    plt.figure(figsize=(8, 5))
    
    # Plot ground-truth
    plt.plot(l_values, gt_inf, 'b-', label='GT Infection (S→I)', linewidth=3)
    plt.plot(l_values, gt_rec, 'r-', label='GT Recovery (I→S)', linewidth=3)
    
    # Plot GNN predictions
    plt.plot(l_values, infection_gnn, 'b--', label='GNN Infection', linewidth=3)
    plt.plot(l_values, recovery_gnn, 'r--', label='GNN Recovery', linewidth=3)
    
    # Plot MLE estimates with error bars
    plt.errorbar(l_values, infection_mle, yerr=np.sqrt(infection_mle*(1-infection_mle)/np.sqrt(total_inf)), 
                 fmt='bo', markersize=8, label='MLE Infection', capsize=5)
    plt.errorbar(l_values, recovery_mle, yerr=np.sqrt(recovery_mle*(1-recovery_mle)/np.sqrt(total_rec)), 
                 fmt='ro', markersize=8, label='MLE Recovery', capsize=5)
    
    # Set labels and plot formatting
    plt.xlabel('Number of Infected Neighbors (ℓ)', fontsize=23, fontweight='bold')
    plt.ylabel('Transition Probability', fontsize=23, fontweight='bold')
    plt.title('Simple Contagion Dynamics', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.ylim(0, 1.05)  # y-axis range
    plt.xlim(0, 10)  # x-axis range
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()  # Clean layout
    plt.show()  # Display plot

# Run the figure generation (this triggers everything above)
generate_figure()
