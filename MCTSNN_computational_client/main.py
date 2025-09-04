import numpy as np

from computational_node_client import ComputationalNodeClient
from mcts_client import AMCTS
from boardPlus import BoardPlus
import torch
from chessModel import ChessNet

device = torch.device("cpu")
print("Using device:", device)
net = ChessNet(80, 30, device=device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
net.load_state_dict(torch.load("C:/Users/micha/Desktop/chess_model/learn_files/chess_model5_epoch5.pt"))

cnd = ComputationalNodeClient()

while True:
    if len(cnd.message_buffer) > 0:
        data = cnd.message_buffer.popleft()
        if data["type"] == "mctsnn":
            policy = np.zeros(6272)
            policy[data["moves"]["id"]] = data["moves"]["probabilities"]
            board = BoardPlus(data["fen"])
            data = AMCTS(200, net).search(board, policy)
            cnd.send(data)
        else:
            cnd.error(f"Unknown message type: {data['type']}")
