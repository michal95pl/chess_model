import os

from boardPlus import BoardPlus
import chess.pgn
import numpy as np
import pickle


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
        real_board = chess.Board()
        moves = []
        boards = []
        wins = []

        for move in game.mainline_moves():
            real_board.push(move)

            if board.changed_perspective:
                move = board.change_move_perspective(move)

            moves.append(board.encode_move(move))
            boards.append(board.encode())
            wins.append(self.white_win(game) * (1 if not board.changed_perspective else -1))

            board.better_push(move)

            #todo: remove in final version
            if not board.changed_perspective and not BoardPlus.compare_boards(board, real_board):
                raise ValueError(
                    f"The encoded board does not match the real board after move. Game: {game.headers['Event']}, Move: {move}")

            board.change_perspective()
        return np.array(moves), np.array(boards), np.array(wins)

    def encode_directory(self, input_path: str, output_path: str, games_in_file: int = 500):
        game_data = []
        i = 0

        for file_name in os.listdir(input_path):
            if file_name.endswith(".pgn"):
                file_name = os.path.join(input_path, file_name)
                pgn = open(file_name)
                print("[INFO] encoding:", file_name)

                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    game_data.append(self.encode_game(game))
                    i += 1
                    if i % games_in_file == 0:
                        game_data_path = f"{output_path}/{i // games_in_file}.rdg"
                        with open(game_data_path, "wb") as f:
                            pickle.dump(game_data, f)
                        game_data = []
                        print("[INFO] saved:", game_data_path)
                pgn.close()

                if game_data:
                    game_data_path = f"{output_path}/{i // games_in_file}.rdg"
                    with open(game_data_path, "wb") as f:
                        pickle.dump(game_data, f)
                    print("[INFO] saved:", game_data_path)
                    game_data = []
