import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import agents
import utils

X : np.ndarray = np.load('data/X.npy')
Y : np.ndarray = np.load('data/Y.npy')

train, val, test = utils.train_val_test_split(X, Y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = torch.jit.load('models/model_full_large_resnet3.pt').to(device)

X_test = torch.from_numpy(test[0]).float()
Y_test = torch.from_numpy(test[1]).float().unsqueeze(1)
test_dataset = data.TensorDataset(X_test, Y_test)
test_loader = data.DataLoader(test_dataset, batch_size=256, shuffle=True)

utils.test_model(model, test_loader)