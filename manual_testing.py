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
1. e4 Nc6 { B00 Nimzowitsch Defense } 2. Bc4 e6 3. Qh5 g6 4. Qf3 f5 5. exf5 gxf5 6. Qh5+ Ke7 7. Qg5+ Nf6 8. d3 Rg8 9. Qh4 d5 10. Bb5 a6 11. Bxc6 bxc6 12. Bh6 Ke8 13. Bxf8 Rxf8 14. Nf3 Rg8 15. Qh6 Rb8 16. b3 Rxg2 17. Ne5 Ng4 18. Qh5+ Ke7 19. Nxc6+ Kd6 20. Nxd8 Nf6 21. Nf7+ Kc6 22. Nd8+ Kd7 23. Qf7+ Kd6 24. Qxf6 Rg6 25. Qf8+ Ke5 26. Nc6+ Kf4 27. Nxb8 Bb7 28. Qb4+ Kg5 29. Qxb7 e5 30. Qxc7 e4 31. dxe4 dxe4 32. Nxa6 h6 33. Rg1+ Kf6 34. Rxg6+ Kxg6 35. b4 Kg5 36. b5 f4 37. b6 e3 38. b7 exf2+ 39. Kxf2 Kg4 40. b8=Q h5 41. Qf7 h4 42. Qbg8+ Kh3 43. Qb3+ Kxh2 44. Qg2# { White wins by checkmate. } 1-0
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = torch.jit.load('models/model_full_large_resnet.pt').to(device)

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