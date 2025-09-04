from log import Log
import socket
from threading import Thread
import json
from collections import deque
import numpy as np


class ComputationalServer(Log):

    def __init__(self):
        super().__init__("MCTSNN_computational_server/log.txt")
        self._connection_timeout = None
        self._port = None
        self._socket = None
        self._max_connections = None
        self.clients = []
        self.clients_buffer = deque()
        try:
            self.__load_config()
        except Exception as e:
            self.error(f"Failed to load configuration: {e}")
            return
        try:
            self.__start_server()
        except Exception as e:
            self.error(f"Failed to start server: {e}")
            return

        Thread(target=self.__get_connections).start()

    def __load_config(self):
        with open("MCTSNN_computational_server/config.json", "r") as file:
            config = json.load(file)
            self._connection_timeout = config["connection_timeout"]
            self._port = config["port"]
            self._max_connections = config["max_connections"]

    def __start_server(self):
        self.info(f"Starting server on port {self._port} with timeout {self._connection_timeout} seconds.")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self._connection_timeout)
        self.socket.bind(('', self._port))
        self.socket.listen(self._max_connections)
        self.info("Server started successfully.")

    def __get_connections(self):
        self.info("Waiting for connections...")
        while True:
            try:
                client_socket, address = self.socket.accept()
                self.info(f"Connection established with {address}.")
                self.clients.append(client_socket)
                Thread(target=self.__receive_message, args=(client_socket,)).start()
            except socket.timeout:
                self.error("Connection timed out, waiting for new connections...")
            except Exception as e:
                self.error(f"Error accepting connection: {e}. Continuing to listen for new connections...")

    def __receive_message(self, client_socket):
        while True:
            try:
                data = client_socket.recv(70656)
                if not data:
                    self.error("Client disconnected.")
                    client_socket.close()
                    self.clients.remove(client_socket)
                    break
                self.clients_buffer.append(json.loads(data.decode('utf-8')))
            except Exception as e:
                self.error(f"Error receiving message: {e}")
                client_socket.close()
                self.clients.remove(client_socket)
                break

    def broadcast_message(self, message):
        for client in self.clients:
            try:
                print(len(json.dumps(message).encode('utf-8')))
                client.sendall(json.dumps(message).encode('utf-8'))
            except Exception as e:
                client.close()
                self.clients.remove(client)
                self.error(f"Failed to send message to a client: {e}")

    def send_mcts_data(self, probs, fen: str):
        data = {
            "type": "mctsnn",
            "moves": {
                "id": [],
                "probabilities": []
            },
            "fen": fen
        }
        for i, prob in enumerate(probs):
            data["moves"]["id"].append(i)
            data["moves"]["probabilities"].append(float(prob))
        self.broadcast_message(data)

    def send_mcts(self, probs, fen):
        pass

    @staticmethod
    def get_moves_quality_from_data(data) -> np.array:
        if data["type"] != "mctsnn":
            raise ValueError("Invalid data type for moves quality.")
        return np.array(data["moves"]["probabilities"])
