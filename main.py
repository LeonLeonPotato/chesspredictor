import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import agents
import utils


learning_rate = 0.0015
weight_decay = 0.0001
batch_size = 512
epochs = 6


X : np.ndarray = np.load('data/X.npy')
Y : np.ndarray = np.load('data/Y.npy')

train, val, test = utils.train_val_test_split(X, Y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = agents.ChessNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()

# torchscript = torch.jit.trace(model, torch.rand(12, 24, 8, 8, device=device))
# torchscript.save('models/model_full_large_resnet2.pt')

X_train = torch.from_numpy(train[0]).float()
Y_train = torch.from_numpy(train[1]).float().unsqueeze(1)

train_dataset = data.TensorDataset(X_train, Y_train)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

X_val = torch.from_numpy(val[0]).float()
Y_val = torch.from_numpy(val[1]).float().unsqueeze(1)
val_dataset = data.TensorDataset(X_val, Y_val)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

utils.train_model(model, criterion, optimizer, train_loader, val_loader, epochs)

torchscript = torch.jit.trace(model, torch.rand(12, 24, 8, 8).to(device))
torchscript.save('models/model_full_large_resnet3.pt')