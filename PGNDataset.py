import os

from boardPlus import BoardPlus
import chess.pgn
import numpy as np
import pickle
from log import Log
import time


class PGNDataset:

    @staticmethod
    def convert_board_to_list(board: BoardPlus):
        """
        Convert board to list (8x8), which contains piece symbols
        """
        board_list = np.full((8, 8), ' ')
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    board_list[i][j] = piece.symbol()
        return board_list

    @staticmethod
    def white_win(game: chess.pgn.Game) -> int:
        """
        Check if white won the game
        """
        result = game.root().headers['Result']
        if result == '1-0' or result == '1/2-1/2':
            return 1
        return -1

    def encode_game(self, game: chess.pgn.Game):
        board = BoardPlus()
        real_board = chess.Board()  # is used to check if the encoded board matches the real board

        moves = []
        boards = []
        wins = []
        for move in game.mainline_moves():
            real_board.push(move)

            if board.changed_perspective:
                move = BoardPlus.change_move_perspective(move)

            moves.append(board.encode_move(move))
            boards.append(board.encode())
            wins.append(self.white_win(game) * (1 if not board.changed_perspective else -1))
            board.better_push(move)

            # todo: remove in final version
            if not board.changed_perspective and not BoardPlus.compare_boards(board, real_board):
                Log.error(
                    "Encoded board does not match the real board. Move: " + str(move) + " at game: " + game.headers[
                        'White'] + " vs " + game.headers['Black'])
                break

            board.change_perspective()
        return np.array(moves), np.array(boards), np.array(wins)

    file_number = 0
    number_converted_games = 0

    @staticmethod
    def save_games_data_to_file(moves: tuple, output_path: str):
        game_data_path = f"{output_path}/{PGNDataset.file_number}.rdg"
        with open(game_data_path, "wb") as f:
            pickle.dump(moves, f)
        PGNDataset.file_number += 1
        Log.info(f"Saved moves to {game_data_path}")
        Log.info(f"Converted total {PGNDataset.number_converted_games} games")

    # problem z metodą concatenate, im większa jest główna lista tym dłużej trwa konkatenacja:
    # [INFO][18:36:29]  100
    # [INFO][18:36:42]  200
    # [INFO][18:37:04] 300
    # [INFO][18:37:33] 400
    # [INFO][18:38:10] 500
    # [INFO][18:38:54] 600
    # [INFO][18:39:45] 700
    # [INFO][18:40:47] 800
    # [INFO][18:41:58] 900
    def encode_directory(self, input_path: str, output_path: str, max_games_in_file: int = 500):
        games_move, games_board, games_win = [], [], []

        for file_name in os.listdir(input_path):
            if file_name.endswith(".pgn"):
                file_name = os.path.join(input_path, file_name)
                pgn = open(file_name)
                Log.info("Converting file: " + file_name)

                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    try:
                        game_moves, game_boards, game_wins = self.encode_game(game)

                        if len(game_moves) == 0 or len(game_boards) == 0 or len(game_wins) == 0:
                            continue
                        games_move.append(game_moves)
                        games_board.append(game_boards)
                        games_win.append(game_wins)

                        PGNDataset.number_converted_games += 1
                        if PGNDataset.number_converted_games % max_games_in_file == 0:
                            moves, boards, wins = np.concatenate(games_move), np.concatenate(games_board), np.concatenate(games_win)
                            PGNDataset.save_games_data_to_file((moves, boards, wins), output_path)
                            games_move, games_board, games_win = [], [], []
                    except Exception as e:
                        Log.error(f"Error processing game: {game.headers}. Error: {e}")
                        continue
                pgn.close()

        if games_move and games_board and games_win:
            moves, boards, wins = np.concatenate(games_move), np.concatenate(games_board), np.concatenate(games_win)
            PGNDataset.save_games_data_to_file((moves, boards, wins), output_path)
