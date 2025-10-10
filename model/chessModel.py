from tqdm import tqdm
import numpy as np
import random
from utils.logger import Logger
import pickle
import torch
import os
import datetime

class ChessModel(Logger):

    def __init__(self, device):
        super().__init__()
        self.device = device

    @staticmethod
    def shuffle_arrays(a, b, c):
        combined = list(zip(a, b, c))
        random.shuffle(combined)
        a_shuffled, b_shuffled, c_shuffled = zip(*combined)
        return np.array(a_shuffled), np.array(b_shuffled), np.array(c_shuffled)


    def train(self, net, optimizer, epochs, data_path = "train_converted_dataset", output_path="learn_files/"):
        file_names = [data_path + '/' + f for f in os.listdir(data_path)]

        if '/' not in output_path:
            self.warning("Learn output path does not contain folder path. Files will be saved in the main project directory.")
        elif not os.path.exists(os.path.dirname(output_path)):
            self.error("Provided folder in learn output path does not exist.")
            return

        for i in range(epochs):
            random.shuffle(file_names)
            policy_losses = []
            value_losses = []

            with tqdm(total=len(file_names), desc=f"Epoch {i}") as pbar:
                for file_name in file_names:
                    try:
                        file_data = pickle.load(open(file_name, "rb"))
                        moves, boards, wins = file_data
                        ChessModel.shuffle_arrays(moves, boards, wins)
                        moves = torch.tensor(moves, device=self.device).float()
                        boards = torch.tensor(boards, device=self.device).float()
                        wins = torch.tensor(wins, device=self.device).float()

                        value, policy = net(boards)
                        value = value.squeeze(1)

                        loss_policy = torch.nn.functional.cross_entropy(policy, moves)
                        loss_value = torch.nn.functional.mse_loss(value, wins)
                        loss = loss_policy + loss_value

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        policy_losses.append(loss_policy.cpu().item())
                        value_losses.append(loss_value.cpu().item())
                        pbar.set_postfix(
                            avg_loss=np.mean(policy_losses + value_losses),
                            avg_policy_loss=np.mean(policy_losses),
                            avg_value_loss=np.mean(value_losses),
                        )
                    except Exception as e:
                        self.error(f"Error processing file {file_name}: {e}")
                        continue
                    finally:
                        pbar.update(1)
            torch.save({
                    "model" : net.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "avg_loss": np.mean(policy_losses + value_losses),
                    "avg_policy_loss": np.mean(policy_losses),
                    "avg_value_loss": np.mean(value_losses),
                    "timestamp": str(datetime.datetime.now())
                },
                output_path + f"_model_epoch{i}.pt"
            )

    """
    Get average loss provided network on data in the given folder path
    """
    @torch.no_grad()
    def get_model_loss(self, net, data_folder_path: str):
        avg_policy_losses = []
        avg_value_losses = []
        net.eval()
        for file_name in os.listdir(data_folder_path):
            file_data = pickle.load(open(os.path.join(data_folder_path, file_name), "rb"))
            moves, boards, wins = file_data
            moves = torch.tensor(moves, device=self.device).float()
            boards = torch.tensor(boards, device=self.device).float()
            wins = torch.tensor(wins, device=self.device).float()

            value, policy = net(boards)
            value = value.squeeze(1)

            loss_policy = torch.nn.functional.cross_entropy(policy, moves)
            loss_value = torch.nn.functional.mse_loss(value, wins)
            avg_policy_losses.append(loss_policy.cpu().item())
            avg_value_losses.append(loss_value.cpu().item())

        return np.mean(avg_policy_losses), np.mean(avg_value_losses)

    @torch.no_grad()
    def get_value_network_confusion_matrix(self, net, test_data_path):
        y_pred_value = []
        y_true_value = []

        y_pred_policy = []
        y_true_policy = []

        with tqdm(total=len(os.listdir(test_data_path)), desc="Calculating confusion matrix", colour="blue") as pbar:
            for file_name in os.listdir(test_data_path):
                file_data = pickle.load(open(os.path.join(test_data_path, file_name), "rb"))
                moves, boards, wins = file_data
                moves = torch.tensor(moves, device=self.device).float()
                boards = torch.tensor(boards, device=self.device).float()
                wins = torch.tensor(wins, device=self.device).float()

                value, policy = net(boards)
                value = value.squeeze(1)
                policy = torch.softmax(policy, dim=1).squeeze(0)

                y_pred_value.append(value.cpu().numpy())
                y_true_value.append(wins.cpu().numpy())

                y_pred_policy.append(policy.cpu().numpy())
                y_true_policy.append(moves.cpu().numpy())
                pbar.update(1)

        y_pred_value = np.concatenate(y_pred_value)
        y_true_value = np.concatenate(y_true_value)
        y_pred_policy = np.concatenate(y_pred_policy)
        y_true_policy = np.concatenate(y_true_policy)

        y_pred_value = np.where(y_pred_value >= 0, 1, 0)
        y_true_value = np.where(y_true_value >= 0, 1, 0)

        y_pred_policy = np.argmax(y_pred_policy, axis=1)
        y_true_policy = np.argmax(y_true_policy, axis=1)

        return (
            y_pred_value,
            y_true_value,
            y_pred_policy,
            y_true_policy
            )