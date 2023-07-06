import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class ChessNet(nn.Module):
#     def __init__(self):
#         super(ChessNet, self).__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(256, 1, kernel_size=3, padding=1),
#             nn.LeakyReLU(),
#         )
#         self.body = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         x = self.head(x)
#         x = x.view(-1, 64)
#         x = self.body(x)
#         return x
    
#     def predict_board(self, x):
#         x = x.reshape(1, 6, 8, 8)
#         x = self.forward(x)
#         return x.item()



class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        x1 = self.convs(x)
        x1 += x
        return F.leaky_relu(x1)

class ChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            *[ResBlock() for _ in range(5)],
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.convs(x)
    
    def predict(self, x):
        x = x.reshape(1, 6, 8, 8)
        return self.forward(x).item()

class NextMovePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            *[ResBlock() for _ in range(5)],
            nn.Flatten(),
            nn.Linear(64*8*8, 128)
        )
    
    def forward(self, x : torch.Tensor):
        nonlegal_to = x[:, -6:, :, :].sum(dim=1, keepdim=True) > 0 # N, 8, 8
        nonlegal_from = x[:, :6, :, :].sum(dim=1, keepdim=True) <= 0 # N, 8, 8
        x = self.convs(x)
        x = x.reshape(-1, 2, 8, 8)
        x = x[:, 0, :, :].masked_fill_(nonlegal_from, float('-inf'))
        x = x[:, 1, :, :].masked_fill_(nonlegal_to, float('-inf'))
        return x