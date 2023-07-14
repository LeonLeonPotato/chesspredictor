import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import io
import chess.pgn

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


def describe(arr):
    print("Array description: ")
    print('  shape', arr.shape)
    print('  mean {:f}'.format(arr.mean()))
    print('  std', arr.std())
    print('  min', arr.min())
    print('  max', arr.max())


def train_val_test_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=1337, shuffle=True)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=1337, shuffle=True)
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

            batch_losses.append(loss.item())

        train_losses.extend(batch_losses)

        val_accuracy = 0
        val_total = 0

        with torch.no_grad():
            model.eval()
            batch_val_losses = []
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                y_val_pred = model(X_val)
                val_loss = loss_fn(y_val_pred, y_val)
                batch_val_losses.append(val_loss.item())

                val_total += torch.flatten(y_val).shape[0]
                val_accuracy += (torch.flatten(y_val_pred).round() == torch.flatten(y_val)).sum().item()
            model.train()

        val_losses.extend(batch_val_losses)
        val_accuracies.append(val_accuracy/val_total)

        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {np.mean(batch_losses):.4f}, Val Loss: {np.mean(batch_val_losses):.4f}, Val Accuracy: {val_accuracy/val_total:.4f}')

        # Loss plot
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train')
        val_plot_x = np.linspace(0, 1, len(val_losses)) * len(train_losses)
        plt.plot(val_plot_x, val_losses, label='Val')
        val_plot_x = np.linspace(0, 1, len(val_accuracies)) * len(train_losses)
        plt.plot(val_plot_x, val_accuracies, label='Val accuracy')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

from sklearn.metrics import mean_squared_error

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    targets = []
    predictions = []

    test_accuracy = 0
    test_total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            
            targets.extend(labels.tolist())
            predictions.extend(outputs.squeeze().tolist())

            #test_accuracy += (torch.sign(outputs).eq(torch.sign(labels))).sum()
            test_accuracy += torch.round(outputs).eq(torch.round(labels)).sum()
            test_total += outputs.shape[0]

    print(f'Test Accuracy: {100*test_accuracy/test_total:.2f}%')
    mse = mean_squared_error(targets, predictions)
    print(f'Test MSE: {mse}')

    print(outputs.T.reshape(-1))
    print(labels.T.reshape(-1))
