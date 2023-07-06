import pandas as pd
import numpy as np
import chess.pgn
import io
import time
from sklearn.utils import resample

df = pd.read_csv('chess_processed_small.csv')
# Assume df is your DataFrame and 'class' is the column with classes

winner_w = df[df['winner'] == 1]
winner_b = df[df['winner'] == -1]
tie = df[df['winner'] == 0]

winner_w = resample(winner_w, 
                    replace=False,    # sample without replacement
                    n_samples=len(winner_b),  # to match minority class
                    random_state=1337) # reproducible results

tie = resample(tie,
                replace=False,    # sample without replacement
                n_samples=len(winner_b),  # to match minority class
                random_state=1337) # reproducible results

df = pd.concat([winner_w, winner_b, tie])

print(df['winner'].value_counts())

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

X = []
Y = []

for idx, row in df.iterrows():
    states = pgn_to_states(row['pgn'])
    winner = row['winner']
    
    laststate = np.zeros((6, 8, 8), dtype=np.int8)
    for state in states:
        X.append(np.concatenate((state, laststate), axis=0))
        Y.append(winner)
        laststate = state

X = np.array(X)
Y = np.array(Y)

np.save('data/X.npy', X)
np.save('data/Y.npy', Y)
