import numpy as np
from boardPlus import BoardPlus
from chessModel import ChessNet
from MCTSNN_computational_server.computional_server import ComputationalServer
import torch
import json


class AMCTS:
    """
    Works only for black perspective.
    """

    def __init__(self, model: ChessNet, server: ComputationalServer):
        self.model = model
        self.server = server

    @torch.no_grad()
    def search(self, state: BoardPlus):
        state = state.__copy__()
        state.change_perspective()  # from black to white

        value, policy = self.model(
            torch.tensor(state.encode(), device=self.model.device).unsqueeze(0).float()
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy *= state.get_available_moves_mask()
        policy /= policy.sum()

        self.server.send_mcts_data(policy, state.fen())

        prob = np.zeros(6272)
        while len(self.server.clients_buffer) == 0:
            pass
        data = self.server.clients_buffer.popleft()
        if data["type"] != "mctsnn":
            raise ValueError("Invalid data type for moves quality.")
        prob[data["moves"]["id"]] = data["moves"]["total_visits"]
        prob /= prob.sum()
        return prob
