import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import agents
import utils
import chess.pgn
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors

import shutil
shutil.rmtree(matplotlib.get_cachedir())

def board_to_array(board):
    board_state = np.zeros((6, 8, 8), dtype=np.int8)

    piece_dict = {
        'P': 0,  # White Pawn
        'R': 1,  # White Rook
        'N': 2,  # White Knight
        'B': 3,  # White Bishop
        'Q': 4,  # White Queen
        'K': 5,  # White King
    }

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(i, j))

            if piece:
                piece_str = str(piece)
                color = int(piece_str.isupper())
                layer = piece_dict[piece_str.upper()]
                board_state[layer, 7-j, i] = color*2-1
            
    return board_state

def pgn_to_states(p):
    game_states = []

    p = io.StringIO(p)
    game = chess.pgn.read_game(p)
    p.close()

    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        board_state = board_to_array(board)
        game_states.append(board_state)

    return game_states

pgn="""
1. e4 e5 2. Bc4 Bc5 3. d3 c6 4. Qe2 d6 5. f4 exf4 6. Bxf4 Qb6
7. Qf3 Qxb2 8. Bxf7+ Kd7 9. Ne2 Qxa1 10. Kd2 Bb4+ 11. Nbc3
Bxc3+ 12. Nxc3 Qxh1 13. Qg4+ Kc7 14. Qxg7 Nd7 15. Qg3 b6
16. Nb5+ cxb5 17. Bxd6+ Kb7 18. Bd5+ Ka6 19. d4 b4 20. Bxb4
Kb5 21. c4+ Kxb4 22. Qb3+ Ka5 23. Qb5# 1-0
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = torch.jit.load('models/model_full_large_resnet4.pt').to(device)

states = pgn_to_states(pgn)
states = torch.from_numpy(np.array(states)).to(device, dtype=torch.float32)

model.eval()
preds = []

a = 0
with torch.inference_mode():
    laststates = [torch.zeros((6, 8, 8), device=device) for i in range(4)]
    for i in states:
        laststates.append(i)
        laststates.pop(0)
        i = torch.stack(laststates, dim=0).float().reshape(1, 6*4, 8, 8)
        y_pred = model(i)
        preds.append(y_pred.item())
        a += (1 - y_pred.item())**2
        #laststate = i[:, -6:, :, :]
        print(y_pred.item())

cmap = plt.get_cmap('coolwarm')

def colorline(x, y1, y2):
    x = np.linspace(x, x + 0.5, 5)
    y = np.linspace(y1, y2, 5)
    for i in range(4):
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=cmap((y[i] + 1) / 2))

for i in range(len(preds) - 1):
    colorline(i / 2, preds[i], preds[i + 1])

plt.ylim(-1, 1)
plt.show()

print("MSE:")
print(a / len(states))
print(a)