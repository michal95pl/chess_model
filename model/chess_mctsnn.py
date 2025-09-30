import numpy as np
from utils.boardPlus import BoardPlus
import math
from model.chessNet import ChessNet
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
        self.lock = False

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

    def __init__(self, sim_count, model: ChessNet, c_param=1.4, max_parallel_computations=1):
        self.sim_count = sim_count
        self.c_param = c_param
        self.model = model
        self.computation_list = []
        self.max_parallel_computations = max_parallel_computations

    def search(self, state: BoardPlus):
        state = state.__copy__()
        state.change_perspective()  # change to black perspective
        root = MCTSNode(state)
        for sim_number in range(self.sim_count):
            best_leaf = self.__selection(root)

            if best_leaf is None:
                self.__make_computation_batch()
                continue
            best_leaf.lock = True

            # end game
            if best_leaf.state.is_game_over():
                result = self.__validate_game(best_leaf.state)
                self.__backpropagation(best_leaf, result)
                best_leaf.lock = False
                continue

            self.computation_list.append(best_leaf)

            if len(self.computation_list) == self.max_parallel_computations or sim_number == self.sim_count - 1:
                self.__make_computation_batch()

        probabilities = np.zeros(state.action_size)
        for child in root.children:
            # if child.total_visits > 0:
                # print(child.total_reward / child.total_visits, child.move)
            probabilities[child.move] = child.total_visits
        probabilities /= probabilities.sum()
        return probabilities

    @torch.no_grad()
    def __make_computation_batch(self):
        encoded_states = np.array([node.state.encode() for node in self.computation_list], dtype=np.float32)
        values, policies = self.model(
            torch.tensor(encoded_states, dtype=torch.float32).to(self.model.device)
        )

        for i in range(len(self.computation_list)):
            value = values[i].item()
            policy = torch.softmax(policies[i], dim=0).cpu().numpy()
            policy *= self.computation_list[i].state.get_available_moves_mask()

            if np.sum(policy) == 0:
                # neural network returned no valid moves
                self.__backpropagation(self.computation_list[i], -1)
                continue
            policy /= policy.sum()

            self.__backpropagation(self.computation_list[i], value)
            self.__expansion(policy, self.computation_list[i])
            self.computation_list[i].lock = False
        self.computation_list.clear()

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
            if child.lock:
                continue
            score = child.get_ucb_score(self.c_param)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def __selection(self, node: MCTSNode):
        while node is not None and not node.is_leaf_node():
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
