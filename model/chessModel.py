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

                        policy_losses.append(loss_policy.item())
                        value_losses.append(loss_value.item())
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


    # def show_5_largest(arr):
    #     indices = np.argpartition(arr, -5)[-5:]
    #     sorted_indices = indices[np.argsort(arr[indices])[::-1]]
    #     largest_values = arr[sorted_indices]
    #     largest_indices = sorted_indices
    #     print(f"Largest values: {largest_values} at indices: {largest_indices}")


    # for manual testing
    # @torch.no_grad()
    # def test(net, optimizer):
    #     net.load_state_dict(torch.load("learn_files/chess_model0_epoch8.pt"))
    #     optimizer.load_state_dict(torch.load("learn_files/chess_optimizer0_epoch8.pt"))
    #
    #     pgn = open("../dataset/lichess_db_standard_rated_2016-03.pgn")
    #
    #     net.eval()
    #     while True:
    #         game = chess.pgn.read_game(pgn)
    #         if game is None:
    #             break
    #         board = BoardPlus(game.board().fen())
    #
    #         for move in game.mainline_moves():
    #
    #             # if board.changed_perspective:
    #             # move = board.change_move_perspective(move)
    #
    #             if board.turn == chess.BLACK:
    #                 prob = chess_mctsnn.AMCTS(1000, net).search(board)
    #                 move_id = np.argmax(prob)
    #                 move_mcts = board.decode_move(move_id)
    #                 move_mcts = board.change_move_perspective(move_mcts)
    #                 print(move_mcts)
    #                 print(move)
    #                 print()
    #             board_tensor = torch.tensor(board.encode(), device=net.device).float().unsqueeze(0)
                # value, policy = net(board_tensor)
                # value = value.item()
                # policy = policy.squeeze(0).cpu().numpy()
                # policy = torch.nn.functional.softmax(torch.tensor(policy), dim=0).numpy()
                #
                # policy *= board.get_available_moves_mask()
                # policy /= np.sum(policy)

                # show_5_largest(board.encode_move(move))
                # show_5_largest(policy)
                # print()

                # board.better_push(move)
                # board.change_perspective()
        # pgn.close()


    # @torch.no_grad()
    # def evaluate(net):
    #     file_names = ["test_converted_dataset/" + f for f in os.listdir("../test_converted_dataset")]
    #     losses = []
    #     policy_losses = []
    #     value_losses = []
    #
    #     with tqdm(total=len(file_names)) as pbar:
    #         for file_name in file_names:
    #             data = pickle.load(open(file_name, "rb"))
    #             try:
    #                 for game_data in data:
    #                     moves, boards, win = game_data
    #
    #                     if len(moves) == 0 or len(boards) == 0 or len(win) == 0:
    #                         continue
    #                     moves = torch.tensor(moves, device=device).float()
    #                     boards = torch.tensor(boards, device=device).float()
    #                     win = torch.tensor(win, device=device).float()
    #
    #                     value, policy = net(boards)
    #                     value = value.squeeze(1)
    #
    #                     loss_policy = torch.nn.functional.cross_entropy(policy, moves)
    #                     loss_value = torch.nn.functional.mse_loss(value, win)
    #                     loss = loss_policy + loss_value
    #
    #                     losses.append(loss.item())
    #                     policy_losses.append(loss_policy.item())
    #                     value_losses.append(loss_value.item())
    #                     pbar.set_postfix(
    #                         avg_loss=np.mean(losses),
    #                         avg_policy_loss=np.mean(policy_losses),
    #                         avg_value_loss=np.mean(value_losses),
    #                     )
    #             except Exception as e:
    #                 print(f"Error evaluating from file {file_name}: {e}")
    #             finally:
    #                 pbar.update(1)

    #