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

class CommandsHandler(Logger):
    def __init__(self):
        super().__init__()
        self.configs = ConfigFile('config.json')
        self.communicationHandler = None
        self.is_model_loaded = False
        self.loaded_model_path = "*"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ChessNet(
            int(self.configs.get_config("num_hidden_layers")),
            int(self.configs.get_config("num_residual_blocks")),
            device=self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def __command_handler(self, command: list):
        if len(command) == 1 and command[0] == 'help':
            self.__help_command()
        elif len(command) == 1 and command[0] == "print-status":
            self.__print_status()
        elif len(command) == 1 and command[0] == 'start-server':
            self.__start_server()
        elif len(command) == 1 and command[0] == 'stop-server':
            self.__stop_server()
        elif len(command) == 4 and command[0] == 'convert-games':
            PGNDataset().encode_directory(command[1], command[2], command[3], int(self.configs.get_config('max_games_per_train_file')), float(self.configs.get_config('test_split_ratio')))
        elif (len(command) == 1 or len(command) == 3) and command[0] == 'train-model':
            if len(command) == 1:
                ChessModel(self.device).train(self.net, self.optimizer, int(self.configs.get_config('epochs')))
            else:
                ChessModel(self.device).train(self.net, self.optimizer, int(self.configs.get_config('epochs')), command[1], command[2])
        elif len(command) == 2 and command[0] == "load-model":
            self.__load_model(command)
        elif (len(command) == 2 or len(command) == 3) and command[0] == "compare-model-to-stockfish":
            self.__compare_model_to_stockfish(command)
        elif (len(command) == 3 or len(command) == 4) and command[0] == "show-loss":
            if len(command) == 3:
                ModelEvaluator(command[1], self.device, self.net).save_losses_plot(command[2])
            else:
                ModelEvaluator(command[1], self.device, self.net).save_losses_plot(command[2], int(command[3]))
            self.is_model_loaded = False
            self.loaded_model_path = "*"
        elif len(command) == 2 and command[0] == "confusion-matrix":
            if not self.is_model_loaded:
                self.warning("Using untrained model. It's recommended to load a trained model before generating confusion matrix.")
            ModelEvaluator(command[1], self.device, self.net).save_confusion_matrix(self.loaded_model_path)
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
        while True:
            command = input("> ").split(' ')
            self.__command_handler(command)

    def __help_command(self):
        print("Available commands:")
        print("print-status")
        print("start-server - Run the chess model server")
        print("stop-server - Stop the chess model server")
        print("convert-games <input_directory> <output_train_data_directory> <output_test_data_directory> - Convert PGN files to encoded format .rdg")
        print("load-model <model_path> - Load a pre-trained model")
        print("train-model <dataset_directory> <output_directory> - Train the model using dataset. dataset and output directory are optional")
        print("compare-model-to-stockfish <num_games> <plot_path> - Evaluate the model against Stockfish. plot_path are optional") #todo: rename
        print("show-loss <test_data_directory> <models_directory> <skip_factor> - Show loss plot for models in models_directory on test data. skip_factor is optional")
        print("confusion-matrix <test_data_directory> - Save confusion matrix for value network on test data")

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
                AMCTS(self.configs.get_config('mcts_simulations'), self.net, self.configs.get_config('mcts_c_param'))
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
            checkpoint = torch.load(command[1], weights_only=False)
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
            AMCTS(self.configs.get_config('mcts_simulations'), self.net, self.configs.get_config('mcts_c_param'))
            self.loaded_model_path,
            int(self.configs.get_config('evaluation_seed'))
        )
        if len(command) == 2:
            eval.evaluate(int(command[1]), int(self.configs.get_config("stockfish_gen_moves")))
        else:
            eval.evaluate(int(command[1]), int(self.configs.get_config("stockfish_gen_moves")), command[2])