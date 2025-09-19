import numpy as np
import chess
import chess.engine
from model.chess_mctsnn import AMCTS
from utils.boardPlus import BoardPlus
import time
from utils.logger import Logger
import matplotlib.pyplot as plt

class StockfishModelEvaluator(Logger):

    def __init__(self, stockfish_path: str, mcts: AMCTS):
        super().__init__()
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.info("Loaded Stockfish engine from " + stockfish_path)
        except Exception as e:
            self.error("Failed to load Stockfish engine: " + str(e))
            raise e
        self.mcts = mcts
        self.random_model_scores = self.__get_random_model_score()

    def evaluate(self, num_games, multipv, save_path: str="evaluation_plot.png"):
        if num_games < 1:
            self.warning("Number of games must be greater than 0")
            return
        self.info("Starting evaluation of " + str(num_games) + " games")
        games_score = []
        for i in range(num_games):
            games_score.append(self.__evaluate_game(multipv))

        self.__create_plot(games_score, save_path)

    def __evaluate_game(self, multipv):
        board = BoardPlus()
        save_time = time.time()
        scores = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = self.engine.analyse(board, chess.engine.Limit(time=0.1), multipv=multipv)
                moves = [info['pv'][0] for info in result]
                move = np.random.choice(moves)
                board.push(move)
            else:
                prob = self.mcts.search(board)
                move_id = np.argmax(prob)
                move = board.decode_move(move_id)
                move = board.change_move_perspective(move)
                board.push(move)
            info = self.engine.analyse(board, chess.engine.Limit(time=0.5))
            score = info['score'].black().score(mate_score=10000)
            scores.append(score)

        self.info("Evaluated game in " + str(time.time() - save_time) + " seconds")
        self.debug("Game result: " + str(board.result()))
        self.debug("Scores during the game: " + str(scores))
        return scores

    def __get_random_model_score(self):
        board = BoardPlus()
        scores = []
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                result = self.engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
            else:
                move = np.random.choice(list(board.legal_moves))
                board.push(move)
            info = self.engine.analyse(board, chess.engine.Limit(time=0.5))
            score = info['score'].black().score(mate_score=10000)
            scores.append(score)
        return scores

    def __create_plot(self, scores, save_path: str):
        # plt.figure(figsize=(10, 6))
        for i, game_scores in enumerate(scores):
            plt.plot(game_scores, label=f'Game {i+1}')
        plt.plot(self.random_model_scores, label='Random Model', linestyle='--', color='black')
        plt.xlabel('Move Number')
        plt.ylabel('Evaluation Score')
        plt.title('Model vs Stockfish')
        plt.legend()
        plt.grid()
        plt.savefig(save_path)
        self.info("Evaluation plot saved to " + save_path)



    def __del__(self):
        try:
            self.engine.quit()
            self.debug("Closed Stockfish engine")
        except Exception as e:
            self.error("Failed to close Stockfish engine: " + str(e))
