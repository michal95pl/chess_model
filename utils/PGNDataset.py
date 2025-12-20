import logging
import os
from utils.boardPlus import BoardPlus
import chess.pgn
import numpy as np
import pickle
from utils.logger import Logger
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from time import sleep
import threading


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
        start_indx = len(board_history) - 1
        stop_indx = start_indx - min(len(board_history), number_of_boards)

        for i in range(start_indx, stop_indx, -1):
            board = board_history[i].__copy__()
            if changed_perspective:
                board.change_perspective()
            temp.append(board.get_board_with_piece_index())

        for _ in range(number_of_boards - len(temp)):
            temp.append(BoardPlus.get_empty_board_with_piece_index())

        return temp

    @staticmethod
    def encode_game(logger, log_lock, game: chess.pgn.Game):
        board = BoardPlus()
        real_board = BoardPlus()

        moves = []
        boards = []
        results = []
        board_history = []

        for move in game.mainline_moves():
            changed_move = move
            if board.changed_perspective:
                changed_move = BoardPlus.change_move_perspective(move)

            moves.append(board.get_move_index(changed_move))
            temp = [board.get_board_with_piece_index()]
            temp.extend(PGNDataset.get_boards_with_piece_index_from_board_history(board_history, board.changed_perspective))
            boards.append(temp)
            results.append(PGNDataset.get_game_result(game, board.changed_perspective))

            board_history.append(real_board.__copy__())

            board.better_push(changed_move)
            real_board.push(move)

            if not board.changed_perspective and not BoardPlus.compare_boards(board, real_board):
                with log_lock:
                    logger.error(f"[PID: {os.getpid()}] Encoded board does not match the real board. Move: {str(move)} at game: {game.headers['White']} vs {game.headers['Black']}")
                break
            board.change_perspective()
        return np.array(moves), np.array(boards), np.array(results)

    @staticmethod
    def __split_moves(moves: tuple, test_split_ratio: float):
        split_index = int(len(moves[0]) * (1 - test_split_ratio))
        return (moves[0][split_index:], moves[1][split_index:], moves[2][split_index:]), (moves[0][:split_index], moves[1][:split_index], moves[2][:split_index])

    @staticmethod
    def shuffle_game_dataset(moves: tuple):
        combined = list(zip(moves[0], moves[1], moves[2]))
        np.random.shuffle(combined)
        combined = [list(t) for t in zip(*combined)]
        return combined[0], combined[1], combined[2]

    @staticmethod
    def __save_games_data_to_file(games: tuple, train_output_path: str, test_output_path: str, test_split_ratio: float, file_counter, shared_game_counter, logger, log_lock):
        shuffle_games = PGNDataset.shuffle_game_dataset(games)
        test_data, train_data = PGNDataset.__split_moves(shuffle_games, test_split_ratio)

        train_moves = np.concatenate(train_data[0])
        train_boards = np.concatenate(train_data[1])
        train_value = np.concatenate(train_data[2])

        if len(test_data[0]) > 0:
            test_moves = np.concatenate(test_data[0])
            test_boards = np.concatenate(test_data[1])
            test_value = np.concatenate(test_data[2])
        else:
            test_moves, test_boards, test_value = [], [], []

        train_data = (train_moves, train_boards, train_value)
        test_data = (test_moves, test_boards, test_value)

        game_train_data_path = f"{train_output_path}/{os.getpid()}_{file_counter}_{len(train_moves)}.rdg"
        game_test_data_path = f"{test_output_path}/{os.getpid()}_{file_counter}_{len(test_moves)}.rdg"

        with open(game_train_data_path, "wb") as f:
            pickle.dump(train_data, f)

        if Logger.log_level == 'DEBUG':
            with log_lock:
                logger.debug(f"[PID: {os.getpid()}] Saved train data moves to {game_train_data_path}")

        if len(test_data[0]) > 0:
            with open(game_test_data_path, "wb") as f:
                pickle.dump(test_data, f)
            if Logger.log_level == 'DEBUG':
                with log_lock:
                    logger.debug(f"[PID: {os.getpid()}] Saved test data moves to {game_test_data_path}")
        else:
            if Logger.log_level == 'DEBUG':
                with log_lock:
                    logger.debug(f"[PID: {os.getpid()}] No test data to save for file {file_counter}.")

        if Logger.log_level == 'DEBUG':
            with log_lock:
                logger.debug(f"Total games encoded: {shared_game_counter.value}")


    @staticmethod
    def encode_directory_worker(shared_game_counter, game_counter_lock, logging_lock, dataset_path: str, files_to_process: list, max_games_in_file:int, test_split_ratio:float, number_of_games:int, train_data_output_path:str, test_data_output_path:str):
        loging = Logger()
        if Logger.log_level == 'DEBUG':
            with logging_lock:
                loging.debug(f"[PID: {os.getpid()}] Started encoding.")

        games_move, games_board, games_win = [], [], []
        file_counter = 0
        game_counter = 0

        for file_name in files_to_process:
            file_name = os.path.join(dataset_path, file_name)
            pgn = open(file_name)

            while True:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break
                try:
                    game_moves, game_boards, game_wins = PGNDataset.encode_game(logging, logging_lock, game)
                    game_counter += 1

                    if len(game_moves) == 0 or len(game_boards) == 0 or len(game_wins) == 0:
                        continue

                    games_move.append(game_moves)
                    games_board.append(game_boards)
                    games_win.append(game_wins)

                    if shared_game_counter.value >= number_of_games:
                        break

                    if game_counter % (max_games_in_file + math.floor(max_games_in_file * 0.1)) == 0:
                        PGNDataset.__save_games_data_to_file((games_move, games_board, games_win), train_data_output_path,
                                                             test_data_output_path, test_split_ratio, file_counter, shared_game_counter, loging, logging_lock)
                        games_move, games_board, games_win = [], [], []
                        file_counter += 1

                    with game_counter_lock:
                        shared_game_counter.value += 1

                except Exception as e:
                    with logging_lock:
                        loging.error(f"[PID: {os.getpid()}] Error while encoding game from file {file_name}: {e}")
                    continue
            pgn.close()

            # break outer loop if limit reached
            if shared_game_counter.value >= number_of_games:
                if Logger.log_level == 'DEBUG':
                    with logging_lock:
                        loging.debug(f"[PID: {os.getpid()}] Reached limit of {number_of_games} games.")
                break

        if games_move:
            PGNDataset.__save_games_data_to_file((games_move, games_board, games_win), train_data_output_path, test_data_output_path,
                                           test_split_ratio, file_counter, shared_game_counter, loging, logging_lock)
            file_counter += 1

        if Logger.log_level == 'DEBUG':
            with logging_lock:
                logging.debug(f"[PID: {os.getpid()}] Finished encoding.")

    @staticmethod
    def _progres_bar(shared_game_counter, number_of_games):
        pbar = tqdm(total=number_of_games, desc="Encoding PGN files", colour="green")
        last_count = 0
        while True:
            current_count = shared_game_counter.value
            pbar.update(current_count - last_count)
            last_count = current_count

            if current_count >= number_of_games:
                break
            sleep(1)

    def encode_directory(self, input_path: str, train_data_output_path: str, test_data_output_path: str, max_games_in_file: int, test_split_ratio: float, number_of_games: int, number_of_workers: int = 16):
        if math.floor(test_split_ratio * max_games_in_file) == 0:
            self.warning("Test split ratio is too small, no test data will be created.")
            self.info(f"Encoding directory with {max_games_in_file} games in each train files")
        else:
            self.info(
                f"Encoding directory with {max_games_in_file} games in each train files and {math.floor(test_split_ratio * max_games_in_file)} games in test files")

        files = [f for f in os.listdir(input_path) if f.endswith('.pgn')]
        tasks = np.array_split(files, number_of_workers)

        manager = multiprocessing.Manager()
        shared_game_counter = manager.Value('i', 0)
        game_counter_lock = manager.Lock()
        logging_lock = manager.Lock()

        with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
            futures = []
            for task in tasks:
                futures.append(
                    executor.submit(
                        PGNDataset.encode_directory_worker,
                        shared_game_counter,
                        game_counter_lock,
                        logging_lock,
                        input_path,
                        list(task),
                        max_games_in_file,
                        test_split_ratio,
                        number_of_games,
                        train_data_output_path,
                        test_data_output_path
                    )
                )

            if Logger.log_level != 'DEBUG':
                bar_thread = threading.Thread(target=PGNDataset._progres_bar, args=(shared_game_counter, number_of_games))
                bar_thread.start()
                bar_thread.join()

            for future in futures:
                future.result()

        self.info("Finished encoding directory.")