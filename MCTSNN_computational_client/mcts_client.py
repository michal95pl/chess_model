import numpy as np
from boardPlus import BoardPlus
import math
from chessModel import ChessNet
import torch


class MCTSNode:
    def __init__(self, state: BoardPlus, parent=None, move=None, prior=0.):
        self.children = []
        self.state = state
        self.parent = parent
        self.move = move
        self.total_visits = 0
        self.total_reward = 0
        self.prior = prior

    def get_ucb_score(self, c_param):
        if self.total_visits == 0:
            exploitation = 0
        else:
            exploitation = self.total_reward / self.total_visits
            exploitation = 1 - (exploitation + 1) / 2

        exploration = c_param * (math.sqrt(self.parent.total_visits) / (1 + self.total_visits)) * self.prior

        return exploitation + exploration

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0


class AMCTS:
    """
    Works only for black perspective.
    """

    def __init__(self, sim_count, model: ChessNet, c_param=1.4):
        self.sim_count = sim_count
        self.c_param = c_param
        self.model = model

    @torch.no_grad()
    def search(self, board: BoardPlus, policy: np.ndarray):
        root = MCTSNode(board)
        self.__expansion(policy, root)

        for _ in range(self.sim_count):
            best_leaf = self.__selection(root)

            # end game
            if best_leaf.state.is_game_over():
                result = self.__validate_game(best_leaf.state)
                self.__backpropagation(best_leaf, result)
            else:
                value, policy = self.model(
                    torch.tensor(best_leaf.state.encode(), device=self.model.device).unsqueeze(0).float()
                )

                value = value.item()
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                policy *= best_leaf.state.get_available_moves_mask()
                if np.sum(policy) == 0:
                    # neural network returned no valid moves
                    self.__backpropagation(best_leaf, -1)
                    continue
                policy /= policy.sum()

                self.__backpropagation(best_leaf, value)
                self.__expansion(policy, best_leaf)

        data = {
            "type": "mctsnn",
            "moves": {
                "id": [],
                "total_visits": []
            },
        }
        for child in root.children:
            data["moves"]["id"].append(child.move)
            data["moves"]["total_visits"].append(child.total_visits)
        return data

    def __validate_game(self, state: BoardPlus):
        winner = state.result()
        if winner == '1-0':
            return 1
        elif winner == '0-1':
            return -1
        return 0

    def __get_best_child(self, node: MCTSNode):
        best_child = None
        best_score = -math.inf
        for child in node.children:
            score = child.get_ucb_score(self.c_param)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def __selection(self, node: MCTSNode):
        while not node.is_leaf_node():
            node = self.__get_best_child(node)
        return node

    def __backpropagation(self, node: MCTSNode, reward):
        while node:
            node.total_reward += reward
            node.total_visits += 1

            node = node.parent
            reward = -reward  # opponent's perspective

    def __expansion(self, policy, node: MCTSNode):
        for move_id in range(len(policy)):
            if policy[move_id] > 0:
                move = node.state.decode_move(move_id)
                board_temp = node.state.__copy__()

                board_temp.better_push(move)
                board_temp.change_perspective()

                new_node = MCTSNode(board_temp, node, move_id, policy[move_id])
                node.children.append(new_node)
