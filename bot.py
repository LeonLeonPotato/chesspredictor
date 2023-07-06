import torch
import chess
import numpy as np
import chess.pgn as pgn
import chess.svg as svg
import cairosvg
import io
import PIL.Image as Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('models/model_full_medium_resnet2.pt').to(device)

board = chess.Board()


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
    game = pgn.read_game(p)
    p.close()

    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        board_state = board_to_array(board)
        game_states.append(board_state)

    return game_states

while not board.is_game_over():
    move, score = None, -99
    for i in board.legal_moves:
        cpboard = board.copy()
        cpboard.push(i)
        inp = torch.from_numpy(np.concatenate((board_to_array(cpboard), board_to_array(board)), axis=0)).float().unsqueeze(0).to(device)
        out = model(inp).item()
        if out > score:
            score = out
            move = i
    board.push(move)
    print(move, score)
    output = io.BytesIO()
    chess_board_svg = svg.board(board)
    cairosvg.svg2png(bytestring=chess_board_svg, write_to=output)
    output.seek(0)
    image = Image.open(output)
    plt.imshow(image)
    plt.axis('off')  # optional, to hide the axis
    plt.show()
    nextmove = input("Enter other's move: ")
    board.push_san(nextmove)
