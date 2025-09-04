import time

import chess_mctsnn
import torch

from chessModel import ChessNet
import numpy as np
import chess
from PGNDataset import PGNDataset
import pickle
from boardPlus import BoardPlus
from chessGUI import chessGUI
import MCTSNN_computational_server.mcts_server as mcts_server

import chess.engine as chess_engine
from MCTSNN_computational_server.computional_server import ComputationalServer
import json

# dataset = PGNDataset()
# dataset.encode_directory("dataset", "F:/train_converted_dataset", 20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print("Using device:", device)
net = ChessNet(80, 30, device=device)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

net.load_state_dict(torch.load("learn_files/chess_model5_epoch5.pt"))
# optimizer.load_state_dict(torch.load("learn_files/chess_optimizer5_epoch5.pt"))
#
board = BoardPlus()
#
# server = ComputationalServer()
#
@torch.no_grad()
def algorithm():

    # prob = mcts_server.AMCTS(net, server).search(board)
    prob = chess_mctsnn.AMCTS(500, net).search(board)
    print()
    move_id = np.argmax(prob)
    move_mcts = board.decode_move(move_id)
    move_mcts = board.change_move_perspective(move_mcts)
    return move_mcts

game = chessGUI(board)
game.add_computer_algorithm_listener(algorithm)
game.run()
