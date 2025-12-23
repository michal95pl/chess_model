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
        self.unobserved_samples = 0

    def get_wu_uct_score(self, c_param):
        if self.total_visits == 0:
            exploitation = 0
        else:
            exploitation = self.total_reward / self.total_visits
            exploitation = 1 - (exploitation + 1) / 2

        exploration = c_param * (math.sqrt(self.parent.total_visits + self.unobserved_samples) / (1 + self.total_visits + self.unobserved_samples)) * self.prior

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
        self.history = []

    def search(self, state: BoardPlus, history_states: list):
        state = state.__copy__()
        state.change_perspective()  # change to black perspective
        self.history = []

        for s in history_states:
            s = s.__copy__()
            s.change_perspective()
            self.history.append(s)

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
            self.__backpropagation_unobserved_samples(best_leaf)

            if len(self.computation_list) == self.max_parallel_computations or sim_number == self.sim_count - 1:
                self.__make_computation_batch()

        probabilities = np.zeros(state.action_size)
        for child in root.children:
            probabilities[child.move] = child.total_visits
        probabilities /= probabilities.sum()
        return probabilities

    @staticmethod
    def get_n_boards_state(node: MCTSNode, state_history, number_of_boards=3):
        boards = []
        perspective = False
        state_history_index = 0
        for _ in range(number_of_boards):
            # above the root
            if node is None:
                if state_history_index < len(state_history):
                    temp_state = state_history[state_history_index].__copy__()
                    if perspective:
                        temp_state.change_perspective()
                    boards.append(temp_state.get_board_with_piece_index())
                    state_history_index += 1
                else:
                    boards.append(BoardPlus.get_empty_board_with_piece_index())
            else:
                temp_state = node.state.__copy__()
                if perspective:
                    temp_state.change_perspective()
                boards.append(temp_state.get_board_with_piece_index())
                node = node.parent
            perspective = not perspective
        return boards

    @torch.no_grad()
    def __make_computation_batch(self):
        boards = np.array([AMCTS.get_n_boards_state(node, self.history, 3) for node in self.computation_list], dtype=np.int32)
        encoded_states = np.eye(13)[boards]
        encoded_states = encoded_states.transpose(0, 1, 4, 2, 3)
        encoded_states = encoded_states.reshape(-1, 39, 8, 8)

        values, policies = self.model(
            torch.tensor(encoded_states, dtype=torch.float32).to(self.model.device)
        )

        wdl = torch.softmax(values, dim=1).cpu().numpy()
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        policies *= np.array([node.state.get_available_moves_mask() for node in self.computation_list])

        for i in range(len(self.computation_list)):
            if np.sum(policies[i]) == 0:
                # neural network returned no valid moves
                self.__backpropagation(self.computation_list[i], -1)
                continue
            policies[i] /= policies[i].sum()
            reward = wdl[i][0] - wdl[i][2]

            self.__backpropagation(self.computation_list[i], reward)
            self.__expansion(self.computation_list[i], policies[i])
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
            score = child.get_wu_uct_score(self.c_param)
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
            node.unobserved_samples = 0

            node = node.parent
            reward = -reward  # opponent's perspective

    def __backpropagation_unobserved_samples(self, node: MCTSNode):
        while node:
            node.unobserved_samples += 1
            node = node.parent

    def __expansion(self, node: MCTSNode, policy: np.ndarray):

        policy_mask = policy > 0
        move_ids = np.arange(len(policy))[policy_mask]
        policy_values = policy[policy_mask]

        for i in range(len(move_ids)):
            move = node.state.decode_move(move_ids[i])
            board_temp = node.state.__copy__()

            board_temp.better_push(move)
            board_temp.change_perspective()

            child_node = MCTSNode(board_temp, node, move_ids[i], policy_values[i])
            node.children.append(child_node)
