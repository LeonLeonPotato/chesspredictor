import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
import agents
import utils
import chess.pgn
import io

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

pgn = """

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
6. d3 b5 7. Bb3 d6 8. a4 Bd7 9. h3 O-O 10. Be3 Na5
11. Ba2 bxa4 12. Nc3 Rb8 13. Bb1 Qe8 14. b3 c5 15. Nxa4 Nc6
16. Nc3 a5 17. Nd2 Be6 18. Nc4 d5 19. exd5 Nxd5 20. Bd2 Nxc3
21. Bxc3 Bxc4 22. bxc4 Bd8 23. Bd2 Bc7 24. c3 f5 25. Re1 Rd8
26. Ra2 Qg6 27. Qe2 Qd6 28. g3 Rde8 29. Qf3 e4 30. dxe4 Ne5
31. Qg2 Nd3 32. Bxd3 Qxd3 33. exf5 Rxe1+ 34. Bxe1 Qxc4 35. Ra1 Rxf5
36. Bd2 h6 37. Qc6 Rf7 38. Re1 Kh7 39. Be3 Be5 40. Qe8 Bxc3
41. Rc1 Rf6 42. Qd7 Qe2 43. Qd5 Bb4 44. Qe4+ Kg8 45. Qd5+ Kh7
46. Qe4+ Rg6 47. Qf5 c4 48. h4 Qd3 49. Qf3 Rf6 50. Qg4 c3
51. Rd1 Qg6 52. Qc8 Rc6 53. Qa8 Rd6 54. Rxd6 Qxd6 55. Qe4+ Qg6
56. Qc4 Qb1+ 57. Kh2 a4 58. Bd4 a3 59. Qc7 Qg6 60. Qc4 c2
61. Be3 Bd6 62. Kg2 h5 63. Kf1 Be5 64. g4 hxg4 65. h5 Qf5
66. Qd5 g3 67. f4 a2 68. Qxa2 Bxf4 0-1
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = torch.jit.load('models/model_full_medium_resnet2.pt').to(device)

states = pgn_to_states(pgn)
states = torch.from_numpy(np.array(states)).to(device, dtype=torch.float32)

model.eval()

a = 0
with torch.no_grad():
    laststate = torch.zeros((1, 6, 8, 8), dtype=torch.float32).to(device)
    for i in states:
        i = i.unsqueeze(0)
        i = torch.cat((laststate, i), dim=1)
        a += (1 - model(i).item())
        laststate = i[:, -6:, :, :]
        print(model(i).item())

print(a / len(states))
print(a)
