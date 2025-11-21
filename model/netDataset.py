from torch.utils.data import Dataset
import numpy as np
import pickle
import random
import torch
import os

class NetDataset(Dataset):
    def __init__(self, converted_data_path):

        self.files_names = [converted_data_path + '/' + f for f in os.listdir(converted_data_path)]

    @staticmethod
    def shuffle_arrays(a, b, c):
        combined = list(zip(a, b, c))
        random.shuffle(combined)
        a_shuffled, b_shuffled, c_shuffled = zip(*combined)
        return np.array(a_shuffled), np.array(b_shuffled), np.array(c_shuffled)

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        file_data = pickle.load(open(self.files_names[idx], "rb"))
        moves, boards, wins = file_data
        moves, boards, wins = NetDataset.shuffle_arrays(moves, boards, wins)
        return torch.from_numpy(moves).float(), torch.from_numpy(boards).float(), torch.from_numpy(wins).float()

    @staticmethod
    def identity_collate_fn(batch_list):
        return batch_list[0]