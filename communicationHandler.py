from threading import Thread
from utils.logger import Logger
from communication import Communication
import time
from utils.boardPlus import BoardPlus
import numpy as np

class CommunicationHandler(Thread, Logger):

    def __init__(self, communication: Communication, mcts):
        Thread.__init__(self)
        Logger.__init__(self)
        self.communication = communication
        self.mcts = mcts
        self.running = True
        self.games_handled = 0
        # self.start()
        self.run()

    def run(self):
        self.info("Waiting for messages")
        while self.running:
            if self.communication.is_message_available():
                message = self.communication.get_first_message()
                conn, command = message
                if command['command'] == 'get_move':
                    board = BoardPlus(command['board'])
                    save_time = time.time()
                    prob = self.mcts.search(board)
                    move_id = np.argmax(prob)
                    move_mcts = board.decode_move(move_id)
                    move_mcts = board.change_move_perspective(move_mcts)
                    self.debug("Computed move: " + str(move_mcts) + " in " + str(round(time.time() - save_time, 2)) + " seconds")
                    self.communication.send({
                        "command": "move",
                        "move": str(move_mcts)
                    }, conn)
                self.games_handled += 1
            time.sleep(1)

    def stop(self):
        self.running = False
        self.communication.close()

    def get_games_handled(self):
        return self.games_handled
