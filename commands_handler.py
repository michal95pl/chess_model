from utils.config_file import ConfigFile
from communication import Communication
from utils.logger import Logger
from utils.PGNDataset import PGNDataset
import torch
from model.chessNet import ChessNet
from model.chessModel import ChessModel
from communicationHandler import CommunicationHandler
from model.chess_mctsnn import AMCTS
from model.stockfish_model_evaluator import StockfishModelEvaluator
from model.model_evaluator import ModelEvaluator
from model.multiprocess_chess_mcts import ParallelAMCTS

class CommandsHandler(Logger):
    def __init__(self):
        super().__init__()
        self.configs = ConfigFile('config.json')
        self.communicationHandler = None
        self.is_model_loaded = False
        self.loaded_model_path = "*"

        self.device = torch.device(("cuda:" + str(self.configs.get_config("gpu_index"))) if torch.cuda.is_available() else "cpu")
        self.net = ChessNet(
            self.device,
            int(self.configs.get_config("num_residual_blocks")),
            int(self.configs.get_config("num_backbone_filters")),
            int(self.configs.get_config("num_policy_filters")),
            int(self.configs.get_config("num_value_filters"))
        )
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-4)

    def __command_handler(self, command: list):
        if command[-1] == '':
            command.pop(-1)

        if len(command) == 1 and command[0] == 'help':
            self.__help_command()
        elif len(command) == 1 and command[0] == "print-status":
            self.__print_status()
        elif len(command) == 1 and command[0] == 'start-server':
            self.__start_server()
        elif len(command) == 1 and command[0] == 'stop-server':
            self.__stop_server()
        elif len(command) == 4 and command[0] == 'convert-games':
            PGNDataset().encode_directory(command[1], command[2], command[3], int(self.configs.get_config('max_games_per_train_file')), float(self.configs.get_config('test_split_ratio')), int(self.configs.get_config('number_of_games')), int(self.configs.get_config('number_of_converting_processes')))
        elif (len(command) <= 3) and command[0] == 'train-model':
            if len(command) == 1:
                ChessModel(self.device).train(self.net, self.optimizer, int(self.configs.get_config('epochs')), int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')), int(self.configs.get_config('buffer_size')))
            elif len(command) == 2:
                ChessModel(self.device).train(self.net, self.optimizer, int(self.configs.get_config('epochs')), int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')), int(self.configs.get_config('buffer_size')), command[1])
            elif len(command) == 3:
                ChessModel(self.device).train(self.net, self.optimizer, int(self.configs.get_config('epochs')), int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')), int(self.configs.get_config('buffer_size')), command[1], command[2])
        elif len(command) == 2 and command[0] == "load-model":
            self.__load_model(command)
        elif (len(command) == 3 or len(command) == 2) and command[0] == "compare-model-to-stockfish":
            self.__compare_model_to_stockfish(command)
        elif (len(command) == 3 or len(command) == 4) and command[0] == "show-loss":
            if len(command) == 3:
                ModelEvaluator(command[1], self.device, self.net).save_losses_plot(command[2], int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')), int(self.configs.get_config('buffer_size')))
            else:
                ModelEvaluator(command[1], self.device, self.net).save_losses_plot(command[2], int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')), int(self.configs.get_config('buffer_size')), int(command[3]))
            self.is_model_loaded = False
            self.loaded_model_path = "*"
        elif len(command) == 2 and command[0] == "confusion-matrix":
            if not self.is_model_loaded:
                self.warning("Using untrained model. It's recommended to load a trained model before generating confusion matrix.")
            ModelEvaluator(command[1], self.device, self.net).save_confusion_matrix(self.loaded_model_path, int(self.configs.get_config('buffer_size')), int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')))
        elif (len(command) == 3 or len(command) == 4) and command[0] == "compare-mcts":
            if not self.is_model_loaded:
                self.warning("Using untrained model. It's recommended to load a trained model before comparing MCTS.")
            self.__compare_mcts(command)
        elif len(command) == 2 and command[0] == "top-k-accuracy":
            if not self.is_model_loaded:
                self.warning("Using untrained model. It's recommended to load a trained model before calculating top-k accuracy.")
            ModelEvaluator(command[1], self.device, self.net).save_top_5_plot(self.loaded_model_path, int(self.configs.get_config('buffer_size')), int(self.configs.get_config('batch_size')), int(self.configs.get_config('number_of_dataset_processes')))
        else:
            print("Unknown command. Type 'help' to see available commands.")


    def run_app(self):
        print(r"      _                                         _      _ ")
        print(r"     | |                                       | |    | |")
        print(r"  ___| |__   ___  ___ ___   _ __ ___   ___   __| | ___| |")
        print(r" / __| '_ \ / _ \/ __/ __| | '_ ` _ \ / _ \ / _` |/ _ \ |")
        print(r"| (__| | | |  __/\__ \__ \ | | | | | | (_) | (_| |  __/ |")
        print(r" \___|_| |_|\___||___/___/ |_| |_| |_|\___/ \__,_|\___|_|")
        print()
        print("Chess Model Application")
        print("Type 'help' to see available commands")

        run_commands = self.configs.get_config("startCommands")
        for cmd in run_commands:
            command = cmd + ' ' + self.configs.get_config2("startCommands", cmd)
            self.info("Executing command: " + command)
            self.__command_handler(command.split(' '))

        while True:
            command = input("> ").split(' ')
            self.__command_handler(command)

    def __help_command(self):
        print("Available commands:")
        print()
        print(" General:")
        print(" - print-status")
        print(" - start-server - Run the chess model server")
        print(" - stop-server - Stop the chess model server")
        print()
        print(" Model and dataset management:")
        print(" - convert-games <input_directory> <output_train_data_directory> <output_test_data_directory> - Convert PGN files to encoded format .rdg")
        print(" - load-model <model_path> - Load a pre-trained model")
        print(" - train-model [dataset_directory] [output_directory] - Train the model using dataset.")
        print()
        print(" Model evaluation:")
        print(" - compare-model-to-stockfish <num_games> [path] - Evaluate the model against Stockfish. Saves evaluation plot and games in path")
        print(" - show-loss <test_data_directory> <models_directory> [skip_factor] - Show loss plot for models in models_directory on test data")
        print(" - confusion-matrix <test_data_directory> - Save confusion matrix for value network on test data")
        print(" - compare-mcts <num_simulations_opponent> <num_simulations_opponent> [path] - Compare the current loaded model with another MCTS. Saves comparison plot and game in path")
        print(" - top-k-accuracy <test_data_directory> - Show top-k accuracy for policy network on test data")

    def __print_status(self):
        print("Computation device: " + str(self.device))
        print("Model loaded: " + str(self.is_model_loaded))
        if self.is_model_loaded:
            print("Loaded model path: " + self.loaded_model_path)
        if self.communicationHandler is not None:
            print("Server is running on " + str(self.configs.get_config('server_ip')) + ":" + str(self.configs.get_config('server_port')))
            print("Games handled in this session: " + str(self.communicationHandler.get_games_handled()))

    def __start_server(self):
        if self.communicationHandler is None:
            if not self.is_model_loaded:
                self.warning("Using untrained model. It's recommended to load a trained model before starting the server.")

            self.communicationHandler = CommunicationHandler(
                Communication(int(self.configs.get_config('server_port')), self.configs.get_config('server_ip')),
                self.__get_mcts()
            )
        else:
            self.info("server is running")

    def __stop_server(self):
        if self.communicationHandler is not None:
            self.communicationHandler.stop()
            self.communicationHandler = None
        else:
            self.info("server is not running")

    def __load_model(self, command: list):
        try:
            checkpoint = torch.load(command[1], weights_only=False, map_location=self.device)
            self.net.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.info("model loaded successfully")
            self.is_model_loaded = True
            self.loaded_model_path = command[1]
        except Exception as e:
            self.error(f"Error loading model or optimizer: {e}")

    def __compare_model_to_stockfish(self, command: list):
        eval = StockfishModelEvaluator(
            self.configs.get_config('stockfish_path'),
            self.__get_mcts(),
            self.loaded_model_path,
            int(self.configs.get_config("stockfish_gen_moves")),
            int(self.configs.get_config('evaluation_seed'))
        )
        if len(command) == 3:
            eval.evaluate(int(command[1]), command[2])
        else:
            eval.evaluate(int(command[1]))

    def __compare_mcts(self, command: list):
        sme = StockfishModelEvaluator(
            self.configs.get_config('stockfish_path'),
            AMCTS(self.configs.get_config('mcts_simulations'), self.net, self.configs.get_config('mcts_c_param'),
                  self.configs.get_config('parallel_computations')),
            self.loaded_model_path,
            int(self.configs.get_config('evaluation_seed'))
        )

        if len(command) == 3:
            sme.compare_to(
                AMCTS(int(command[1]), self.net, self.configs.get_config('mcts_c_param'),
                      int(command[2]))
            )
        else:
            sme.compare_to(
                AMCTS(int(command[1]), self.net, self.configs.get_config('mcts_c_param'),
                      int(command[2])),
                command[3]
            )

    def __get_mcts(self):
        if bool(self.configs.get_config("multiprocessing_mcts")):
            return ParallelAMCTS(self.configs.get_config('mcts_simulations'), self.net, self.configs.get_config('mcts_c_param'), self.configs.get_config('number_of_processes'))
        else:
            return AMCTS(self.configs.get_config('mcts_simulations'), self.net, self.configs.get_config('mcts_c_param'), self.configs.get_config('parallel_computations')),