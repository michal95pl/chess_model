import os

from sympy.physics.units.definitions.unit_definitions import statampere

from utils.boardPlus import BoardPlus
import chess.pgn
import numpy as np
import pickle
from utils.logger import Logger
import math


class PGNDataset(Logger):

    def __init__(self):
        super().__init__()

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
    def get_game_result(game: chess.pgn.Game, changed_perspective: bool) -> np.ndarray:
        """
        Check if white won the game
        """
        result = game.root().headers['Result']
        temp = np.zeros(3, dtype=np.int8)

        if changed_perspective:
            if result == '1-0':
                temp[2] = 1
            elif result == '0-1':
                temp[0] = 1
        else:
            if result == '1-0':
                temp[0] = 1
            elif result == '0-1':
                temp[2] = 1

        if result == '1/2-1/2':
            temp[1] = 1

        return temp

    @staticmethod
    def get_boards_with_piece_index_from_board_history(board_history: list, changed_perspective, number_of_boards: int = 2):
        temp = []
        number_empty_boards = number_of_boards - len(board_history) if len(board_history) < number_of_boards else 0

        for _ in range(number_empty_boards):
            temp.append(BoardPlus.get_empty_board_with_piece_index())

        for b in board_history[-(number_of_boards - number_empty_boards):]:
            board = b.__copy__()
            if changed_perspective:
                board.change_perspective()
            temp.append(board.get_board_with_piece_index())

        return temp
    def encode_game(self, game: chess.pgn.Game):
        board = BoardPlus()
        real_board = BoardPlus()  # is used to check if encoded board is correct and for history

        moves = []
        boards = []
        results = []
        board_history = []

        for move in game.mainline_moves():
            changed_move = move
            if board.changed_perspective:
                changed_move = BoardPlus.change_move_perspective(move)

            moves.append(board.get_move_index(changed_move))
            temp = PGNDataset.get_boards_with_piece_index_from_board_history(board_history, board.changed_perspective)
            temp.append(board.get_board_with_piece_index())
            boards.append(temp)
            results.append(PGNDataset.get_game_result(game, board.changed_perspective))

            board_history.append(real_board.__copy__())

            board.better_push(changed_move)
            real_board.push(move)

            if not board.changed_perspective and not BoardPlus.compare_boards(board, real_board):
                self.error(
                    "Encoded board does not match the real board. Move: " + str(move) + " at game: " + game.headers[
                        'White'] + " vs " + game.headers['Black'])
                break
            board.change_perspective()
        return np.array(moves), np.array(boards), np.array(results)

    file_number = 0
    number_converted_games = 0

    @staticmethod
    def __split_moves(moves: tuple, test_split_ratio: float):
        split_index = int(len(moves) * (1 - test_split_ratio))
        return (moves[0][split_index:], moves[1][split_index:], moves[2][split_index:]), (moves[0][:split_index], moves[1][:split_index], moves[2][:split_index])

    @staticmethod
    def shuffle_game_dataset(moves: tuple):
        combined = list(zip(moves[0], moves[1], moves[2]))
        np.random.shuffle(combined)
        combined = [list(t) for t in zip(*combined)]
        return combined[0], combined[1], combined[2]

    def __save_games_data_to_file(self, moves: tuple, train_output_path: str, test_output_path: str, test_split_ratio: float):
        game_train_data_path = f"{train_output_path}/{PGNDataset.file_number}.rdg"
        game_test_data_path = f"{test_output_path}/{PGNDataset.file_number}.rdg"

        data = PGNDataset.shuffle_game_dataset(moves)
        train_data, test_data = PGNDataset.__split_moves(data, test_split_ratio)

        with open(game_train_data_path, "wb") as f:
            pickle.dump(train_data, f)
            self.info(f"Saved train data moves to {game_train_data_path}")

        if len(test_data[0]) > 0:
            with open(game_test_data_path, "wb") as f:
                pickle.dump(test_data, f)
                self.info(f"Saved test data moves to {game_test_data_path}")

        PGNDataset.file_number += 1
        self.info(f"Converted total {PGNDataset.number_converted_games} games")

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
    def encode_directory(self, input_path: str, train_data_output_path: str, test_data_output_path: str, max_games_in_file: int, test_split_ratio: float, number_of_games: int):
        games_move, games_board, games_win = [], [], []

        if math.floor(test_split_ratio * max_games_in_file) == 0:
            self.warning("Test split ratio is too small, no test data will be created.")
            self.info(f"Encoding directory with {max_games_in_file} games in each train files")
        else:
            self.info(f"Encoding directory with {max_games_in_file} games in each train files and {math.floor(test_split_ratio * max_games_in_file)} games in test files")

        for file_name in os.listdir(input_path):
            if file_name.endswith(".pgn"):
                file_name = os.path.join(input_path, file_name)
                pgn = open(file_name)
                self.info("Converting file: " + file_name)

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

                        if PGNDataset.number_converted_games >= number_of_games:
                            self.info(f"Reached the limit of {number_of_games} games. Stopping conversion.")
                            break

                        if PGNDataset.number_converted_games % (max_games_in_file + math.floor(max_games_in_file*0.1)) == 0:
                            moves, boards, wins = np.concatenate(games_move), np.concatenate(games_board), np.concatenate(games_win)
                            self.__save_games_data_to_file((moves, boards, wins), train_data_output_path, test_data_output_path, test_split_ratio)
                            games_move, games_board, games_win = [], [], []
                    except Exception as e:
                        self.error(f"Error processing game: {game.headers}. Error: {e}")
                        continue
                pgn.close()

                # break outer loop if limit reached
                if PGNDataset.number_converted_games >= number_of_games:
                    break

        if games_move and games_board and games_win:
            moves, boards, wins = np.concatenate(games_move), np.concatenate(games_board), np.concatenate(games_win)
            self.__save_games_data_to_file((moves, boards, wins), train_data_output_path, test_data_output_path, test_split_ratio)
