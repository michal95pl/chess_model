from torch.utils.data import IterableDataset
import numpy as np
import pickle
import random
import torch
import os

class NetDataset(IterableDataset):
    def __init__(self, converted_data_path, buffer_size=1000):
        self.files_names = [converted_data_path + '/' + f for f in os.listdir(converted_data_path)]
        self.number_of_moves = 0
        for file_name in self.files_names:
            file_inf = file_name.split('_')
            self.number_of_moves += int(file_inf[-1].split('.')[0])
        self.buffer_size = buffer_size

    @staticmethod
    def shuffle_arrays(a, b, c):
        combined = list(zip(a, b, c))
        random.shuffle(combined)
        a_shuffled, b_shuffled, c_shuffled = zip(*combined)
        return np.array(a_shuffled), np.array(b_shuffled), np.array(c_shuffled)

    def __iter__(self):

        proc_inf = torch.utils.data.get_worker_info()
        files = self.files_names.copy()

        if proc_inf is not None:
            random.seed(proc_inf.seed)
            worker_id = proc_inf.id
            num_workers = proc_inf.num_workers
            files = files[worker_id::num_workers] # divide files among workers

        random.shuffle(files)
        buffer = []

        for file_name in files:
            file_data = pickle.load(open(file_name, "rb"))
            moves, boards, wins = file_data

            buffer.extend(zip(moves, boards, wins))

            while len(buffer) >= self.buffer_size:
                i = random.randint(0, len(buffer) - 1)
                yield self.__encode(buffer[i])

                buffer[i] = buffer[-1]
                buffer.pop()

        random.shuffle(buffer)
        for item in buffer:
            yield self.__encode(item)

    def __encode(self, buf_data):
        move, board, win = buf_data
        move = torch.tensor(move).long()

        board = np.eye(13)[board]
        board = board.transpose(0, 3, 1, 2)
        board = board.reshape(-1, 8, 8)
        board = torch.from_numpy(board).float()

        win = torch.tensor(win).float()

        return move, board, win

    def __len__(self):
        return self.number_of_moves