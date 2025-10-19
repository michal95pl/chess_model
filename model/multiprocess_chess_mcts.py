import numpy as np
from utils.boardPlus import BoardPlus
import math
from model.chessNet import ChessNet
import torch
from utils.logger import Logger
from multiprocessing import shared_memory, Value, Lock
from concurrent.futures import ProcessPoolExecutor

tree_dtype = np.dtype([
    ('parent_id', np.int32), # -1 for root
    ('children', np.uint32, 100), # avg max children per node is ~(30-55)
    ('children_count', np.uint8),
    ('fen', 'S100'),
    ('changed_perspective', np.bool_),
    ('move_id', np.int16), # -1 for root
    ('total_visit', np.uint16),
    ('total_reward', np.float32),
    ('prior', np.float32),
    ('unobserved_samples', np.uint16),
    ('is_locked', np.bool_),
    ('result', np.int8) # result for terminated states. 2 for not terminated
])

last_index_lock_g = None
back_propagation_lock_g = None
def init_worker(last_index_lock, back_propagation_lock):
    global last_index_lock_g
    global back_propagation_lock_g
    last_index_lock_g = last_index_lock
    back_propagation_lock_g = back_propagation_lock

class ParallelAMCTS(Logger):
    """
    Works only for black perspective.
    """

    def __init__(self, sim_count, model: ChessNet, c_param=1.4, max_parallel_computations=1):
        super().__init__()
        self.sim_count = sim_count
        self.c_param = c_param
        self.model = model
        self.max_parallel_computations = max_parallel_computations
        self.computation_list = []

        self.tree = np.zeros(700000, dtype=tree_dtype)
        self.tree_shm = shared_memory.SharedMemory(create=True, size=self.tree.nbytes)
        self.tree = np.ndarray(self.tree.shape, dtype=self.tree.dtype, buffer=self.tree_shm.buf)
        self.debug(f"MCTS tree allocated with size: {self.tree.nbytes / 1e6}MB")

        self.last_index = np.zeros(1, dtype=np.int32)
        self.last_index_shm = shared_memory.SharedMemory(create=True, size=self.last_index.nbytes)
        self.last_index = np.ndarray(self.last_index.shape, dtype=self.last_index.dtype, buffer=self.last_index_shm.buf)

        self.last_index_lock = Lock()
        self.back_propagation_lock = Lock()

        self.process_executor = ProcessPoolExecutor(max_workers=16, initializer=init_worker, initargs=(self.last_index_lock, self.back_propagation_lock))
        self.debug("Created 16 process for MCTS computations.")

    def __del__(self):
        self.tree_shm.close()
        self.tree_shm.unlink()
        self.last_index_shm.close()
        self.last_index_shm.unlink()
        self.process_executor.shutdown(wait=True)
        self.debug("Shut down MCTS")

    def search(self, state: BoardPlus):
        state = state.__copy__()
        state.change_perspective() # change to black perspective

        self.last_index[0] = 0
        self.tree.fill(0)
        ParallelAMCTS.add_node(self.tree, self.last_index_lock, self.last_index, state) # add root node

        for sim_number in range(self.sim_count):
            best_leaf_id = self.__selection(0) # selection from root node

            if best_leaf_id is None:
                self.__make_computation_batch()
                continue

            if self.is_terminated(best_leaf_id):
                result = self.tree[best_leaf_id]['result']
                ParallelAMCTS.__backpropagation(self.tree, best_leaf_id, result, self.back_propagation_lock)
                continue

            self.tree[best_leaf_id]['is_locked'] = True
            self.computation_list.append(best_leaf_id)
            self.__backpropagation_unobserved_samples(best_leaf_id)

            if len(self.computation_list) == self.max_parallel_computations or sim_number == self.sim_count - 1:
                self.__make_computation_batch()

        probabilities = np.zeros(state.action_size)
        for child_id in self.tree[0]['children'][:self.tree[0]['children_count']]:
            probabilities[self.tree[child_id]['move_id']] = self.tree[child_id]['total_visit']
        probabilities /= probabilities.sum()
        return probabilities

    @torch.no_grad()
    def __make_computation_batch(self):
        # convert to objects

        computation_states = []
        for node_id in self.computation_list:
            fen = bytes(self.tree[node_id]['fen']).rstrip(b'\x00').decode('utf-8')
            board_state = BoardPlus(fen=fen)
            board_state.changed_perspective = self.tree[node_id]['changed_perspective']
            computation_states.append(board_state)

        encoded_states = np.array([state.encode() for state in computation_states], dtype=np.float32)
        values, policies = self.model(
            torch.tensor(encoded_states, dtype=torch.float32).to(self.model.device)
        )

        values = values.detach().squeeze(-1).cpu().numpy()
        policies = torch.softmax(policies, dim=1).cpu().numpy()
        policies *= np.array([state.get_available_moves_mask() for state in computation_states])

        tasks = [(policies[i], values[i], self.computation_list[i], self.tree_shm.name, self.tree.shape, tree_dtype, self.last_index_shm.name) for i in range(len(self.computation_list))]
        futures = [self.process_executor.submit(ParallelAMCTS.computation_worker, *task) for task in tasks]

        for future in futures:
            future.result()
        self.computation_list.clear()

    @staticmethod
    def computation_worker(policy, value, node_id, tree_shm_name, tree_shape, tree_dtype, last_index_shm_name):
        global last_index_lock_g
        global back_propagation_lock_g
        # print(os.getpid() , "a")
        shm = shared_memory.SharedMemory(name=tree_shm_name)
        tree = np.ndarray(tree_shape, dtype=tree_dtype, buffer=shm.buf)

        shm_last_index = shared_memory.SharedMemory(name=last_index_shm_name)
        last_index = np.ndarray((1,), dtype=np.int32, buffer=shm_last_index.buf)

        if np.sum(policy) == 0:
            # neural network returned no valid moves
            ParallelAMCTS.__backpropagation(tree, node_id, -1, back_propagation_lock_g)
            return
        policy /= policy.sum()

        ParallelAMCTS.__backpropagation(tree, node_id, value, back_propagation_lock_g)
        ParallelAMCTS.__expansion(tree, node_id, policy, last_index, back_propagation_lock_g)
        tree[node_id]['is_locked'] = False

    @staticmethod
    def __validate_game(state: BoardPlus):
        winner = state.result()
        if winner == '1-0':
            return 1
        elif winner == '0-1':
            return -1
        return 0

    def __get_best_child(self, node_id):
        best_child_id = None
        best_score = -math.inf
        for child_id in self.tree[node_id]['children'][:self.tree[node_id]['children_count']]:
            if self.tree[child_id]['is_locked']:
                continue
            score = self.get_wu_uct_score(child_id, self.c_param)
            if score > best_score:
                best_score = score
                best_child_id = child_id
        return best_child_id

    def __selection(self, node_id):
        # root node is locked
        if self.tree[node_id]['is_locked']:
            return None
        # node_id in none when all children are locked
        while node_id is not None and not self.is_leaf_node(node_id):
            node_id = self.__get_best_child(node_id)
        return node_id

    @staticmethod
    def __backpropagation(tree, node_id, reward, lock):
        while node_id != -1:
            with lock:
                tree[node_id]['total_reward'] += reward
                tree[node_id]['total_visit'] += 1
                tree[node_id]['unobserved_samples'] = 0
            node_id = tree[node_id]['parent_id']
            reward = -reward  # opponent's perspective

    def __backpropagation_unobserved_samples(self, node_id):
        while node_id != -1:
            self.tree[node_id]['unobserved_samples'] += 1
            node_id = self.tree[node_id]['parent_id']

    @staticmethod
    def __expansion(tree, node_id, policy: np.ndarray, last_index, lock):

        policy_mask = policy > 0
        move_ids = np.arange(len(policy))[policy_mask]
        policy_values = policy[policy_mask]

        fen = bytes(tree[node_id]['fen']).rstrip(b'\x00').decode('utf-8')
        board_state = BoardPlus(fen=fen)
        board_state.changed_perspective = tree[node_id]['changed_perspective']

        for i in range(len(move_ids)):
            move = board_state.decode_move(move_ids[i])
            temp_board = board_state.__copy__()

            temp_board.better_push(move)
            temp_board.change_perspective()

            tree[node_id]['children'][tree[node_id]['children_count']] = ParallelAMCTS.add_node(
                    tree= tree,
                    lock=lock,
                    last_index=last_index,
                    state= temp_board,
                    parent_id=node_id,
                    move_id=move_ids[i],
                    prior=policy_values[i],
                    result= ParallelAMCTS.__validate_game(temp_board) if temp_board.is_game_over() else 2
                )
            tree[node_id]['children_count'] += 1
        # print("childrens: ", tree[node_id]['children_count'], " for node id ", node_id)

    def get_wu_uct_score(self, id_node, c_param):
        if self.tree[id_node]['total_visit'] == 0:
            exploitation = 0
        else:
            exploitation = self.tree[id_node]['total_reward'] / self.tree[id_node]['total_visit']
            exploitation = 1 - (exploitation + 1) / 2

        parent_id = self.tree[id_node]['parent_id']
        exploration = c_param * (math.sqrt(self.tree[parent_id]['total_visit'] + self.tree[id_node]['unobserved_samples']) / (
                1 + self.tree[id_node]['total_visit'] + self.tree[id_node]['unobserved_samples'])) * self.tree[id_node]['prior']

        return exploitation + exploration

    def is_terminated(self, id_node) -> bool:
        return self.tree[id_node]['result'] != 2

    def is_leaf_node(self, id_node) -> bool:
        return self.tree[id_node]['children_count'] == 0

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
        tree[node_id]['result'] = result # to improve performance (in main loop it avoid creating additional BoardPlus objects)
        return node_id
