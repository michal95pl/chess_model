from MCTSNN import AMCTS
from TicTacToe import Game
import numpy as np
import torch
import random
from tqdm import tqdm
from model import Net

class AlphaZero:
    def __init__(self, optimizer, model: Net, game: Game):
        self.optimizer = optimizer
        self.model = model
        self.iteration = 10
        self.epoch = 10
        self.self_play_iter = 500
        self.batch_size = 64
        self.number_of_threads = 2
        self.game = game
        self.mcts = AMCTS(60, model)

        # temperature for exploration. <1, ...)
        # higher value means more exploration. Flattens the distribution
        self.temperature = 1.30

    def self_play(self):
        memory = []
        state = self.game.__copy__()
        player = 1

        while True:
            prob_move = self.mcts.search(state)

            memory.append((state, prob_move, player))

            # make move
            prob_move = prob_move ** (1 / self.temperature)
            prob_move = prob_move / prob_move.sum()
            move = np.random.choice(state.get_all_actions(), p=prob_move)
            state.push(move, 1)

            if state.is_terminated():
                # convert to real player perspective
                result = state.get_win_result() * player

                out_memory = []
                for mem_state, mem_prob_move, mem_player in memory:
                    if mem_player == player:
                        tmp_result = result
                    else:
                        tmp_result = mem_state.get_oponent_value(result)

                    out_memory.append((mem_state.encode_board(mem_state.board), mem_prob_move, tmp_result))
                return out_memory

            state.switch_to_opponent_perspective()
            player = Game.get_oponent(player)

    def train(self, memory):
        random.shuffle(memory)

        loses = []
        for batch_index in range(0, len(memory), self.batch_size):
            batch = memory[batch_index: min(batch_index + self.batch_size, len(memory) - 1)]
            states, probs, rewards = zip(*batch)

            states, probs, rewards = np.array(states), np.array(probs), np.array(rewards).reshape(-1, 1)

            states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
            probs = torch.tensor(probs, dtype=torch.float32, device=self.model.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)

            out_value, out_policy = self.model(states)

            loss_policy = torch.nn.functional.cross_entropy(out_policy, probs)
            loss_value = torch.nn.functional.mse_loss(out_value, rewards)
            loss = loss_policy + loss_value
            loses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(loses)

    def learn(self):
        for iteration in range(self.iteration):
            memory = []

            self.model.eval()
            for _ in tqdm(range(self.self_play_iter)):
                memory += self.self_play()

            self.model.train()
            for epoch in range(self.epoch):
                loss = self.train(memory)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            torch.save(self.model.state_dict(), f"learn_files/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"learn_files/optimizer_{iteration}.pt")
