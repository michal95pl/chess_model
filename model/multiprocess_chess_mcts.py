import time

import numpy as np
from utils.boardPlus import BoardPlus
from model.chessNet import ChessNet
from utils.logger import Logger
from multiprocessing import shared_memory, Value, Lock
from concurrent.futures import ProcessPoolExecutor
from model.tree_computation_worker import TreeComputationWorker

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

selection_lock_g = None
backpropagation_lock_g = None
expansion_lock_g = None
computation_worker_g = None
def init_worker(selection_lock, backpropagation_lock, expansion_lock, model, c_param):
    print("init")
    global selection_lock_g
    global backpropagation_lock_g
    global expansion_lock_g
    global computation_worker_g

    selection_lock_g = selection_lock
    backpropagation_lock_g = backpropagation_lock
    expansion_lock_g = expansion_lock

    model.to('cuda')
    model.eval()
    computation_worker_g = TreeComputationWorker(model, c_param)

class ParallelAMCTS(Logger):
    """
    Works only for black perspective.
    """

    def __init__(self, sim_count, model: ChessNet, c_param=1.4, max_parallel_computations=1):
        super().__init__()
        self.sim_count = sim_count

        self.tree = np.zeros(2000000, dtype=tree_dtype)
        self.tree_shm = shared_memory.SharedMemory(create=True, size=self.tree.nbytes)
        self.tree = np.ndarray(self.tree.shape, dtype=self.tree.dtype, buffer=self.tree_shm.buf)
        self.debug(f"MCTS tree allocated with size: {self.tree.nbytes / 1e6}MB")

        self.last_index = np.zeros(1, dtype=np.int32)
        self.last_index_shm = shared_memory.SharedMemory(create=True, size=self.last_index.nbytes)
        self.last_index = np.ndarray(self.last_index.shape, dtype=self.last_index.dtype, buffer=self.last_index_shm.buf)

        self.selection_lock = Lock()
        self.backpropagation_lock = Lock()
        self.expansion_lock = Lock()

        self.process_executor = ProcessPoolExecutor(max_workers=10, initializer=init_worker, initargs=(self.selection_lock, self.backpropagation_lock, self.expansion_lock, model, c_param))
        self.debug("Created 4 process for MCTS computations.")
        self.debug("Warming up MCTS processes...")

        futures = []
        for _ in range(10):
            futures.append(self.process_executor.submit(
                ParallelAMCTS.warmup_process
            ))

        for future in futures:
            future.result()
        self.debug("MCTS processes warmed up.")

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

        # add root node
        self.last_index[0] = 0
        TreeComputationWorker.add_node(self.tree, self.selection_lock, self.last_index, state) # add root node

        # start simulations
        futures = []
        sims_per_process = self.sim_count // 10
        for _ in range(10):
            s_time = time.time()
            futures.append(self.process_executor.submit(
                ParallelAMCTS.run_process,
                sims_per_process,
                self.tree_shm.name,
                self.tree.shape,
                self.tree.dtype,
                self.last_index_shm.name
            ))
            print(time.time() - s_time)

        for future in futures:
            future.result()
        print(self.tree[0])
        probabilities = np.zeros(state.action_size)
        for child_id in self.tree[0]['children'][:self.tree[0]['children_count']]:
            probabilities[self.tree[child_id]['move_id']] = self.tree[child_id]['total_visit']
        probabilities /= probabilities.sum()
        return probabilities

    @staticmethod
    def run_process(sim_count, tree_name, tree_shape, tree_dtype, last_index_name):
        global selection_lock_g
        global backpropagation_lock_g
        global expansion_lock_g
        global computation_worker_g
        print("Process started simulations.")
        computation_worker_g.compute_tree(sim_count, tree_name, tree_shape, tree_dtype, last_index_name,
                                          selection_lock_g, backpropagation_lock_g)

    @staticmethod
    def warmup_process():
        time.sleep(10)