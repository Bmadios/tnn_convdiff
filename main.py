from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from tnn import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Début du chronomètre
start_time = time.time()

import pandas as pd

# Load data
data = pd.read_csv("solution_u_implicit_data.csv")

def truncate_two_decimals(value):
    return int(value * 100) / 100.0

def truncate_three_decimals(value):
    return int(value * 1000) / 1000.0

#u_at_0_003 = data[data["temps"] == 0.003]["u"].values

# Étape 2: Remplacez les valeurs u aux timesteps 0, 0.001 et 0.002 par cette valeur
#for t in [0, 0.001, 0.002]:
    #data.loc[data["temps"] == t, "u"] = u_at_0_003
# Fonction pour vérifier si un nombre est un multiple de 10^-2
def is_multiple_of_10_minus_2(value, tolerance=1e-4):
    return abs(value * 100 - round(value * 100)) < tolerance

#print(data['temps'].unique())

data["temps"] = data["temps"].apply(truncate_three_decimals)
# Filtrer les données pour ne garder que celles dont le temps est un multiple de 10^-2
data = data[data["temps"].apply(is_multiple_of_10_minus_2)]
#print(data['temps'].unique())

data["x"] = data["x"].apply(truncate_two_decimals)
data["y"] = data["y"].apply(truncate_two_decimals)
#data["temps"] = data["temps"].apply(truncate_three_decimals)
data["u"] = data["u"].apply(truncate_three_decimals)


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i+sequence_length][['temps', 'x', 'y']].values
        target = data.iloc[i+sequence_length]['u']
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

sequence_length = 10  # Hyperparameter
sequences, targets = create_sequences(data, sequence_length)



# Split data into training and remaining data
train_seq, remaining_seq, train_target, remaining_target = train_test_split(sequences, targets, test_size=0.8, random_state=42)

# Split remaining data into validation and test
val_seq, test_seq, val_target, test_target = train_test_split(remaining_seq, remaining_target, test_size=0.5, random_state=42)


train_seq_tensor = torch.tensor(train_seq, dtype=torch.float32)
train_target_tensor = torch.tensor(train_target, dtype=torch.float32)

val_seq_tensor = torch.tensor(val_seq, dtype=torch.float32)
val_target_tensor = torch.tensor(val_target, dtype=torch.float32)

test_seq_tensor = torch.tensor(test_seq, dtype=torch.float32)
test_target_tensor = torch.tensor(test_target, dtype=torch.float32)

# Create TensorDatasets
train_data = TensorDataset(train_seq_tensor, train_target_tensor)
val_data = TensorDataset(val_seq_tensor, val_target_tensor)
test_data = TensorDataset(test_seq_tensor, test_target_tensor)

# Create DataLoaders
batch_size = 2  # This is a hyperparameter
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

model = TimeSeriesTransformer(input_dim=3, d_model=64, nhead=4, num_encoder_layers=3, dim_feedforward=256)
model.to(device)  # Move to GPU if available

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 10  # Hyperparameter

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        src, target = batch
        src, target = src.to(device), target.to(device)
        output = model(src).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")




