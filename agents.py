import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess.pgn

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

class MoveProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            *[ResBlock() for _ in range(4)],
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return self.convs(x)

class ChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.move_processor = MoveProcessor()
        self.convs = nn.Sequential(
            nn.Conv2d(64*4, 64, kernel_size=3, padding=1),
            *[ResBlock() for _ in range(1)],
            nn.Flatten(),
            nn.Linear(64*8*8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        newx = []
        for i in range(4):
            newx.append(self.move_processor(x[:, i*6:(i+1)*6, :, :]))
        x = torch.cat(newx, dim=1)
        return self.convs(x)
    
    def predict(self, x):
        x = x.reshape(1, 6, 8, 8)
        return self.forward(x).item()
    
class ChessAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chessnet = ChessNet()
    
    def forward(self, x):
        return self.chessnet(x)
    
    def board_to_tensor(self, board: chess.Board):
        piece_map = board.piece_map()
        tensor = torch.zeros(12, 8, 8)
        for pos, piece in piece_map.items():
            layer = (piece.color * 6) + piece.piece_type - 1
            row, col = divmod(pos, 8)
            tensor[layer, row, col] = 1
        board.move_stack
        return tensor

    def predict(self, board: chess.Board):
        tensor_list = []
        clone = board.copy()
        
        for _ in range(3):
            tensor = self.board_to_tensor(clone)
            tensor_list.append(tensor)
            if len(clone.move_stack) > 0:
                clone.pop()  # Revert the last move

        input_tensor = torch.stack(tensor_list[::-1]).unsqueeze(0)  # Reverse to get chronological order

        if board.turn == chess.BLACK:
            input_tensor = torch.flip(input_tensor, [2]).mul(-1)

        
        return self.forward(input_tensor).item()