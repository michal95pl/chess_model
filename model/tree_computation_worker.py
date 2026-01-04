import numpy as np
from utils.boardPlus import BoardPlus
import math
from model.chessNet import ChessNet
import torch
from multiprocessing import shared_memory
import time

class TreeComputationWorker:

    def __init__(self, model: ChessNet, c_param: float):
        self.model = model
        self.c_param = c_param
        self.computation_list = []

    def compute_tree(self, num_simulations: int, tree_shm_name, tree_shape, tree_dtype, last_index_shm_name, tree_stats_lock, tree_id_lock, fen_history: list):

        shm = shared_memory.SharedMemory(name=tree_shm_name)
        tree = np.ndarray(tree_shape, dtype=tree_dtype, buffer=shm.buf)

        shm_last_index = shared_memory.SharedMemory(name=last_index_shm_name)
        last_index = np.ndarray((1,), dtype=np.int32, buffer=shm_last_index.buf)

        self.computation_list = [] # it's not necessary, but to be sure it's empty before starting computation

        sim_num = 0
        while sim_num < num_simulations:
            with tree_stats_lock:
                best_leaf_id = self.__selection(tree)
                if best_leaf_id is not None:
                    TreeComputationWorker.backpropagate_stats(tree, best_leaf_id)
                    tree[best_leaf_id]['is_locked'] = True

            if best_leaf_id is None:
                # process any pending computations to unlock dead nodes
                if len(self.computation_list) > 0:
                    self.__make_computation_batch(tree, tree_stats_lock, tree_id_lock, last_index, fen_history)
                    self.computation_list.clear()
                else:
                    # wait for other processes to unlock nodes
                    time.sleep(0.1)
                continue

            if TreeComputationWorker.is_terminated(tree, best_leaf_id):
                result = tree[best_leaf_id]['result']
                TreeComputationWorker.__backpropagation(tree, best_leaf_id, result, tree_stats_lock)
                sim_num += 1
                tree[best_leaf_id]['is_locked'] = False
                continue

            self.computation_list.append(best_leaf_id)

            if len(self.computation_list) == 2 or sim_num == num_simulations - 1:
                self.__make_computation_batch(tree, tree_stats_lock, tree_id_lock, last_index, fen_history)
                self.computation_list.clear()
            sim_num += 1

    @staticmethod
    def backpropagate_stats(tree, id_node):
        while id_node != -1:
            tree[id_node]['unobserved_samples'] += 1
            id_node = tree[id_node]['parent_id']

    def get_n_boards_state(self, tree, node_id, fen_history, number_of_boards=3):
        boards = []
        perspective = False
        fen_history_index = 0
        for _ in range(number_of_boards):
            # above the root
            if node_id == -1:
                if fen_history_index < len(fen_history):
                    fen = bytes(fen_history[fen_history_index]).rstrip(b'\x00').decode('utf-8')
                    state = BoardPlus(fen=fen)
                    if perspective:
                        state.change_perspective()
                    boards.append(state.get_board_with_piece_index())
                    fen_history_index += 1
                else:
                    boards.append(BoardPlus.get_empty_board_with_piece_index())
            else:
                fen = bytes(tree[node_id]['fen']).rstrip(b'\x00').decode('utf-8')
                board_state = BoardPlus(fen=fen)
                board_state.changed_perspective = tree[node_id]['changed_perspective']
                if perspective:
                    board_state.change_perspective()
                boards.append(board_state.get_board_with_piece_index())
                node_id = tree[node_id]['parent_id']
            perspective = not perspective
        return boards

    @torch.no_grad()
    def __make_computation_batch(self, tree, tree_stats_lock, tree_id_lock, last_index, fen_history):
        boards = []
        available_moves_masks = []
        states = []
        for node_id in self.computation_list:
            boards.append(self.get_n_boards_state(tree, node_id, fen_history))
            state = BoardPlus(fen=bytes(tree[node_id]['fen']).rstrip(b'\x00').decode('utf-8'))
            state.changed_perspective = tree[node_id]['changed_perspective']
            available_moves_masks.append(state.get_available_moves_mask())
            states.append(state)

        encoded_states = np.eye(13)[boards]
        encoded_states = encoded_states.transpose(0, 1, 4, 2, 3)
        encoded_states = encoded_states.reshape(-1, 39, 8, 8)

        values, policies = self.model(
            torch.tensor(encoded_states, dtype=torch.float32).to(self.model.device)
        )

        wdl = torch.softmax(values, dim=1).cpu().numpy()
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        policies *= np.array(available_moves_masks)

        for i, node_id in enumerate(self.computation_list):
            value = wdl[i][0] - wdl[i][2]  # win prob - loss prob
            TreeComputationWorker.__backpropagation(tree, node_id, value, tree_stats_lock)

            if np.sum(policies[i]) == 0:
                tree[node_id]['is_locked'] = False
                continue

            policies[i] /= np.sum(policies[i])
            self.__expansion(tree, states[i], node_id, policies[i], last_index, tree_id_lock)
            tree[node_id]['is_locked'] = False

    @staticmethod
    def __backpropagation(tree, node_id, reward, lock):
        with lock:
            while node_id != -1:
                tree[node_id]['total_reward'] += reward
                tree[node_id]['unobserved_samples'] -= 1
                tree[node_id]['total_visit'] += 1
                node_id = tree[node_id]['parent_id']
                reward = -reward  # opponent's perspective

    @staticmethod
    def get_wu_uct_score(tree, id_node, c_param):
        parent_id = tree[id_node]['parent_id']

        if tree[id_node]['total_visit'] == 0:
            # FPU. We try to find worse case than ours for opponent
            exploitation = tree[parent_id]['total_reward'] / tree[parent_id]['total_visit']
            exploitation = (exploitation + 1) / 2 # scale to [0, 1]
        else:
            exploitation = tree[id_node]['total_reward'] / tree[id_node]['total_visit']
            exploitation = 1 - (exploitation + 1) / 2 # reverse and scale to [0, 1]. Reverse to weaken opponent's moves

        exploration = c_param * (math.sqrt(tree[parent_id]['total_visit'] + tree[id_node]['unobserved_samples']) / (
                1 + tree[id_node]['total_visit'] + tree[id_node]['unobserved_samples'])) * tree[id_node]['prior']
        return exploitation + exploration

    # return None when root is locked or all children are locked
    def __selection(self, tree):
        node_id = 0
        # root node is locked
        if tree[node_id]['is_locked']:
            return None

        # node_id in none when all children are locked
        while node_id is not None and not TreeComputationWorker.is_leaf_node(tree, node_id):
            node_id = self.get_best_child(tree, node_id)
        return node_id

    @staticmethod
    def __expansion(tree, board_state, node_id, policy: np.ndarray, last_index, lock):

        policy_mask = policy > 0
        move_ids = np.arange(len(policy))[policy_mask]
        policy_values = policy[policy_mask]

        for i in range(len(move_ids)):
            move = board_state.decode_move(move_ids[i])
            temp_board = board_state.__copy__()

            temp_board.push(move)
            temp_board.change_perspective()

            tree[node_id]['children'][tree[node_id]['children_count']] = TreeComputationWorker.add_node(
                tree=tree,
                lock=lock,
                last_index=last_index,
                state=temp_board,
                parent_id=node_id,
                move_id=move_ids[i],
                prior=policy_values[i],
                result=TreeComputationWorker.__validate_game(temp_board) if temp_board.is_game_over() else 2
            )
            tree[node_id]['children_count'] += 1

    def get_best_child(self, tree, node_id):
        best_child_id = None
        best_score = -math.inf
        for child_id in tree[node_id]['children'][:tree[node_id]['children_count']]:
            if tree[child_id]['is_locked']:
                continue
            score = TreeComputationWorker.get_wu_uct_score(tree, child_id, self.c_param)
            if score > best_score:
                best_score = score
                best_child_id = child_id
        return best_child_id

    @staticmethod
    def is_terminated(tree, id_node) -> bool:
        return tree[id_node]['result'] != 2

    @staticmethod
    def is_leaf_node(tree, id_node) -> bool:
        return tree[id_node]['children_count'] == 0

    @staticmethod
    def __validate_game(state: BoardPlus):
        winner = state.result()
        if winner == '1-0':
            return 1
        elif winner == '0-1':
            return -1
        return 0

    @staticmethod
    def add_node(tree, lock, last_index, state: BoardPlus, parent_id=-1, move_id=-1, prior=0.0, result=2):
        with lock:
            node_id = last_index[0]
            last_index[0] += 1

        tree[node_id]['fen'] = state.fen().encode('utf-8')
        tree[node_id]['changed_perspective'] = state.changed_perspective
        tree[node_id]['parent_id'] = parent_id
        tree[node_id]['move_id'] = move_id
        tree[node_id]['prior'] = prior
        tree[node_id]['total_visit'] = 0
        tree[node_id]['total_reward'] = 0.0
        tree[node_id]['unobserved_samples'] = 0
        tree[node_id]['is_locked'] = False
        tree[node_id]['children_count'] = 0
        tree[node_id]['result'] = result  # to improve performance (it avoid creating additional BoardPlus objects in main loop)
        return node_id
