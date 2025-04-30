import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from torch_geometric.utils import from_networkx

# Load COVID-19 data
df = pd.read_csv('india_cases.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Select and preprocess features
features = ['new_deaths', 'new_cases_per_million', 'new_deaths_per_million', 'reproduction_rate', 'people_vaccinated']

# Replace empty/0 values appropriately
df['new_deaths'] = df['new_deaths'].fillna(1)
df.loc[df['new_deaths'] == 0, 'new_deaths'] = 1

df['new_cases_per_million'] = df['new_cases_per_million'].fillna(1)
df.loc[df['new_cases_per_million'] == 0, 'new_cases_per_million'] = 1

df['new_deaths_per_million'] = df['new_deaths_per_million'].fillna(1)
df.loc[df['new_deaths_per_million'] == 0, 'new_deaths_per_million'] = 1

df['reproduction_rate'] = df['reproduction_rate'].fillna(0.0001)
df.loc[df['reproduction_rate'] == 0, 'reproduction_rate'] = 0.0001

df['people_vaccinated'] = df['people_vaccinated'].fillna(0)
df.loc[df['people_vaccinated'] == 0, 'people_vaccinated'] = 0

data = df[features].values

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Convert to time series format - predicting all features
def create_sequences(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])  # Predict all features
    return np.array(X), np.array(Y)

window_size = 14
X, Y = create_sequences(scaled_data, window_size)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")

# Create a synthetic population network (Germany has ~83M people, we'll use n=100 for demonstration)
n = 100  # Number of nodes in our population graph
m = 2    # Number of edges to attach from new node to existing nodes
G = nx.barabasi_albert_graph(n, m)
edge_index = from_networkx(G).edge_index
adj = torch.tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float32)

# Graph Attention Layer
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

# GAT Model
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
    def __init__(self, input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads, n_nodes):
        super(RNN_GAT_Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc1 = torch.nn.Linear(hidden_size, n_nodes * hidden_size)  # Expand to graph nodes
        self.gat = GAT(hidden_size, gat_hidden, output_size, num_heads)
        self.fc2 = torch.nn.Linear(n_nodes * output_size, output_size)  # Compress back to features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.n_nodes = n_nodes
        self.output_size = output_size

    def forward(self, x, adj):
        # RNN processing
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        rnn_out, _ = self.rnn(x, h0)  # [batch_size, seq_len, hidden_size]
        rnn_out = rnn_out[:, -1, :]  # Take last time step [batch_size, hidden_size]
        
        # Expand to graph nodes
        node_features = self.fc1(rnn_out).view(-1, self.n_nodes, self.hidden_size)  # [batch_size, n_nodes, hidden_size]
        
        # GAT processing
        gat_out = self.gat(node_features, adj)  # [batch_size, n_nodes, output_size]
        
        # Aggregate and compress
        gat_out = gat_out.reshape(gat_out.size(0), -1)  # Flatten node features
        output = self.fc2(gat_out)  # [batch_size, output_size]
        
        return output

# Model parameters
input_size = 5  # Number of features
hidden_size = 128
output_size = 5  # Predicting all features
num_layers = 3
gat_hidden = 64
num_heads = 4
n_nodes = n  # Number of nodes in population graph
epochs = 800
lr = 0.0005
train_ratio = 0.5

# Split into training and testing sets
train_size = int(len(X) * train_ratio)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Initialize model
model = RNN_GAT_Model(input_size, hidden_size, output_size, num_layers, gat_hidden, num_heads, n_nodes)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

# Training loop with early stopping
losses = []
val_losses = []
best_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    Y_pred_train = model(X_train, adj)
    loss = criterion(Y_pred_train, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        Y_pred_val = model(X_test, adj)
        val_loss = criterion(Y_pred_val, Y_test).item()
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Testing phase
model.eval()
with torch.no_grad():
    Y_pred_test = model(X_test, adj)
    test_loss = criterion(Y_pred_test, Y_test).item()
print(f"Final Test Loss: {test_loss:.6f}")

# Inverse transform predictions for all features
Y_train_actual = scaler.inverse_transform(Y_train.numpy())
Y_pred_train_actual = scaler.inverse_transform(Y_pred_train.detach().numpy())
Y_test_actual = scaler.inverse_transform(Y_test.numpy())
Y_pred_test_actual = scaler.inverse_transform(Y_pred_test.detach().numpy())

# Plot results for each feature
feature_names = ['New Deaths', 'New Cases per Million', 'New Deaths per Million', 
                'Reproduction Rate', 'People Vaccinated']
plt.figure(figsize=(20, 15))

for i, feature in enumerate(features):
    plt.subplot(3, 2, i+1)
    
    # Training data plot
    train_dates = df['date'][window_size:train_size+window_size]
    plt.plot(train_dates, Y_train_actual[:, i], label=f'Actual {feature_names[i]} (Train)', color='blue', alpha=0.5)
    plt.plot(train_dates, Y_pred_train_actual[:, i], label=f'Predicted {feature_names[i]} (Train)', color='red', linestyle='--')
    
    # Testing data plot
    test_dates = df['date'][train_size+window_size:]
    plt.plot(test_dates, Y_test_actual[:, i], label=f'Actual {feature_names[i]} (Test)', color='green', alpha=0.5)
    plt.plot(test_dates, Y_pred_test_actual[:, i], label=f'Predicted {feature_names[i]} (Test)', color='orange', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel(feature_names[i])
    plt.title(f'{feature_names[i]} Prediction')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()
plt.grid()
plt.show()










# Plot results for each feature
feature_names = ['New Deaths', 'New Cases per Million', 'New Deaths per Million', 
                 'Reproduction Rate', 'People Vaccinated']
plt.figure(figsize=(20, 15))

for i, feature in enumerate(features):
    plt.subplot(3, 2, i + 1)
    
    # Training data plot
    train_dates = df['date'][window_size:train_size + window_size]
    plt.plot(train_dates, Y_train_actual[:, i], label=f'Actual {feature_names[i]} (Train)', color='blue', alpha=0.5)
    plt.plot(train_dates, Y_pred_train_actual[:, i], label=f'Predicted {feature_names[i]} (Train)', color='red', linestyle='--')
    
    # Testing data plot
    test_dates = df['date'][train_size + window_size:]
    plt.plot(test_dates, Y_test_actual[:, i], label=f'Actual {feature_names[i]} (Test)', color='green', alpha=0.5)
    plt.plot(test_dates, Y_pred_test_actual[:, i], label=f'Predicted {feature_names[i]} (Test)', color='orange', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel(feature_names[i])
    plt.title(f'{feature_names[i]} Prediction')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
