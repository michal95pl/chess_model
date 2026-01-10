import os
import numpy as np
import chess
import chess.engine
import chess.pgn
import torch
from utils.PGNDataset import PGNDataset
from utils.boardPlus import BoardPlus
import time
from utils.logger import Logger
import matplotlib.pyplot as plt

class StockfishModelEvaluator(Logger):

    def __init__(self, stockfish_path: str, model_name: str, multipv, seed: int = 42):
        super().__init__()
        self.model_name = model_name
        self.rng = np.random.default_rng(seed)
        self.multipv = multipv
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.info("Loaded Stockfish engine from " + stockfish_path)
        except Exception as e:
            self.error("Failed to load Stockfish engine: " + str(e))
            raise e

    def evaluate(self, mcts, num_games, path:str = "results"):
        if not os.path.exists(path):
            self.error("Provided folder path does not exist.")
            return
        if num_games < 1:
            self.error("Number of games must be greater than 0")
            return

        self.info(f"Evaluating random model")
        random_model_scores = self.__get_random_model_score()
        self.info("Starting evaluation of " + str(num_games) + " games")
        games_score = []
        for i in range(num_games):
            games_score.append(self.__evaluate_game(mcts, path + f"/game{i+1}.pgn"))
        self.__create_mcts_stockfish_plot(mcts, random_model_scores, games_score, path + "/evaluation_plot.png")

    def evaluate_model(self, num_games, model, path:str = "results"):
        if not os.path.exists(path):
            self.error("Provided folder path does not exist.")
            return
        if num_games < 1:
            self.error("Number of games must be greater than 0")
            return

        self.info(f"Evaluating random model")
        random_model_scores = self.__get_random_model_score()
        self.info("Starting evaluation of " + str(num_games) + " games")
        games_score = []
        for i in range(num_games):
            games_score.append(self.__evaluate_model(path + f"/game{i+1}.pgn", model))
        self.__create_net_stockfish_plot(random_model_scores, games_score, path + "/evaluation_plot.png")

    def compare_to(self, mcts, opponent_mcts, save_path: str= "results"):
        if not os.path.exists(save_path):
            self.error("Provided folder path does not exist.")
            return
        self.info("Starting comparison game")
        board = BoardPlus()
        game = chess.pgn.Game()
        node = game

        scores = []
        state_history = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:

                temp_history = []
                for state in StockfishModelEvaluator.get_last_states(state_history):
                    temp = state.__copy__()
                    temp.change_perspective()
                    temp_history.append(temp)

                temp_board = board.__copy__()
                temp_board.change_perspective()
                prob = opponent_mcts.search(temp_board, temp_history)
                move_id = np.argmax(prob)
                move = temp_board.decode_move(move_id)
            else:
                prob = mcts.search(board, StockfishModelEvaluator.get_last_states(state_history))
                move_id = np.argmax(prob)
                move = board.decode_move(move_id)
                move = board.change_move_perspective(move)
            board.push(move)
            state_history.append(board.__copy__())
            node = node.add_variation(move)

            if not board.is_game_over():
                info = self.engine.analyse(board, chess.engine.Limit(time=1))
                score = info['score'].black().score(mate_score=10000)
                scores.append(score)
        self.__create_comparison_plot(scores, opponent_mcts, save_path + "/comparison_plot.png")
        with open(save_path + '/comparison_game.pgn', 'w') as pgn_file:
            pgn_file.write(str(game))
        self.info("Saved comparison game to " + save_path + "/comparison_game.pgn")

    @staticmethod
    def get_last_states(history, n_max=2):
        states = []
        i = len(history) - 1
        for _ in range(n_max):
            if i < 0:
                return states
            states.append(history[i])
            i -= 1
        return states

    def __evaluate_game(self, mcts, game_path):
        board = BoardPlus()
        game = chess.pgn.Game()
        node = game

        save_time = time.time()
        scores = []
        computation_times = []
        state_history = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = self.engine.analyse(board, chess.engine.Limit(time=0.1), multipv=self.multipv)
                moves = [info['pv'][0] for info in result]
                move = self.rng.choice(moves)
            else:
                c_save_time = time.time()
                prob = mcts.search(board, StockfishModelEvaluator.get_last_states(state_history))
                move_id = np.argmax(prob)
                move = board.decode_move(move_id)
                move = board.change_move_perspective(move)
                computation_times.append(time.time() - c_save_time)
            board.push(move)
            state_history.append(board.__copy__())
            node = node.add_variation(move)

            info = self.engine.analyse(board, chess.engine.Limit(time=0.5))
            score = info['score'].black().score(mate_score=10000)
            scores.append(score)

        self.info("Evaluated game in " + str(time.time() - save_time) + " seconds")
        self.debug("Game result: " + str(board.result()))
        self.debug("Scores during the game: " + str(scores))
        self.debug("Computation time: " + str(computation_times))

        with open(game_path, 'w') as pgn_file:
            pgn_file.write(str(game))
        self.info("Saved game to " + game_path)

        return scores, np.mean(computation_times)

    @torch.no_grad()
    def __evaluate_model(self, game_path, model):
        board = BoardPlus()
        game = chess.pgn.Game()
        node = game

        save_time = time.time()
        scores = []
        computation_times = []
        state_history = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = self.engine.analyse(board, chess.engine.Limit(time=0.1), multipv=self.multipv)
                moves = [info['pv'][0] for info in result]
                move = self.rng.choice(moves)
            else:
                c_save_time = time.time()
                encoded_states = PGNDataset.get_boards_with_piece_index_from_board_history(state_history, True, 3)
                encoded_states = np.eye(13)[encoded_states]
                encoded_states = encoded_states.transpose(0, 3, 1, 2)
                encoded_states = encoded_states.reshape(39, 8, 8)
                values, policies = model(
                    torch.tensor(encoded_states, dtype=torch.float32).to(model.device).unsqueeze(0)
                )
                policy = torch.softmax(policies, dim=1).cpu().numpy()[0]
                temp_board = board.__copy__()
                temp_board.change_perspective()

                policy *= np.array(temp_board.get_available_moves_mask())
                policy /= policy.sum()
                move_id = np.argmax(policy)
                move = board.decode_move(move_id)
                move = board.change_move_perspective(move)

                computation_times.append(time.time() - c_save_time)
            board.push(move)
            state_history.append(board.__copy__())
            node = node.add_variation(move)

            info = self.engine.analyse(board, chess.engine.Limit(time=0.5))
            score = info['score'].black().score(mate_score=10000)
            scores.append(score)

        self.info("Evaluated game in " + str(time.time() - save_time) + " seconds")
        self.debug("Game result: " + str(board.result()))
        self.debug("Scores during the game: " + str(scores))
        self.debug("Computation time: " + str(computation_times))

        with open(game_path, 'w') as pgn_file:
            pgn_file.write(str(game))
        self.info("Saved game to " + game_path)

        return scores, np.mean(computation_times)

    def __get_random_model_score(self):
        board = BoardPlus()
        scores = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = self.engine.analyse(board, chess.engine.Limit(time=0.1), multipv=self.multipv)
                moves = [info['pv'][0] for info in result]
                move = self.rng.choice(moves)
            else:
                move = np.random.choice(list(board.legal_moves))
            board.push(move)
            info = self.engine.analyse(board, chess.engine.Limit(time=0.5))
            score = info['score'].black().score(mate_score=10000)
            scores.append(score)
        return scores

    def __create_mcts_stockfish_plot(self, mcts, random_model_scores, scores, save_path: str):
        # plt.figure(figsize=(10, 6))
        games_scores, computation_times = zip(*scores)
        for i, game_scores in enumerate(games_scores):
            plt.plot(game_scores, label=f'Game {i+1}')
        plt.plot(random_model_scores, label='Random Model', linestyle='--', color='black')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation Score')
        plt.title(
            'Model vs Stockfish\n' +
            f'Model: {self.model_name}\n' +
            f"MCTS: sim: {mcts.sim_count}, c: {mcts.c_param}\n" +
            f'Avg computation time (s): {round(np.mean(computation_times), 2)}'
        )
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        self.info("Evaluation plot saved to " + save_path)

    def __create_net_stockfish_plot(self, random_model_scores, scores, save_path: str):
        # plt.figure(figsize=(10, 6))
        games_scores, computation_times = zip(*scores)
        for i, game_scores in enumerate(games_scores):
            plt.plot(game_scores, label=f'Game {i+1}')
        plt.plot(random_model_scores, label='Random Model', linestyle='--', color='black')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation Score')
        plt.title(
            'Model vs Stockfish\n' +
            f'Model: {self.model_name}\n' +
            f'Avg computation time (s): {round(np.mean(computation_times), 2)}'
        )
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        self.info("Evaluation plot saved to " + save_path)

    def __create_comparison_plot(self, scores, mcts_opponent, save_path: str):
        plt.figure(figsize=(10, 6))
        plt.plot(scores)
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation Score')
        plt.title(
            'Model vs op model\n' +
            f'Model: {self.model_name}\n' +
            f"MCTS: sim: {self.mcts.sim_count}, c: {self.mcts.c_param}, parallel: {self.mcts.max_parallel_computations}\n" +
            f'Opponent MCTS: sim: {mcts_opponent.sim_count}, c: {mcts_opponent.c_param}, parallel: {mcts_opponent.max_parallel_computations}'
        )
        # plt.legend()
        plt.grid()
        plt.savefig(save_path)
        self.info("Evaluation plot saved to " + save_path)


    def __del__(self):
        try:
            self.engine.quit()
            self.debug("Closed Stockfish engine")
        except Exception as e:
            self.error("Failed to close Stockfish engine: " + str(e))
