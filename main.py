
import chess_mctsnn
import torch

from chessModel import ChessNet
import numpy as np
import chess
from PGNDataset import PGNDataset
import pickle
from boardPlus import BoardPlus
from chessGUI import chessGUI

import chess.engine as chess_engine

# dataset = PGNDataset()
# dataset.encode_directory("dataset", "F:/train_converted_dataset", 20)

device = torch.device("cpu")
print("Using device:", device)
net = ChessNet(80, 30, device=device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

net.load_state_dict(torch.load("learn_files/chess_model3_epoch5.pt"))
optimizer.load_state_dict(torch.load("learn_files/chess_optimizer3_epoch5.pt"))

board = BoardPlus()

@torch.no_grad()
def algorithm():
    prob = chess_mctsnn.AMCTS(100, net).search(board)
    move_id = np.argmax(prob)
    move_mcts = board.decode_move(move_id)
    move_mcts = board.change_move_perspective(move_mcts)

    return move_mcts


game = chessGUI(board)
game.add_computer_algorithm_listener(algorithm)
game.run()
