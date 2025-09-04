from log import Log
import json
import socket
from boardPlus import BoardPlus
from collections import deque
from threading import Thread


class ComputationalNodeClient(Log):
    CONFIG_FILE = "config.json"

    def __init__(self):
        super().__init__("log.txt")
        self._sock = None
        self._ip = None
        self._port = None
        self._connection_timeout = None
        self.message_buffer = deque()
        try:
            self.__load_config()
        except Exception as e:
            self.error(f"Failed to load configuration: {e}")
            return
        try:
            self.__connect()
        except Exception as e:
            self.error(f"Failed to connect to the server: {e}")
            return


    def __load_config(self):
        with open(ComputationalNodeClient.CONFIG_FILE, "r") as f:
            data = json.load(f)
            self._ip = data["host"]["ip"]
            self._port = data["host"]["port"]
            self._connection_timeout = data["connection_timeout"]

    def __connect(self):
        self.info(f"Connecting to server at {self._ip}:{self._port} with timeout {self._connection_timeout}ms")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.sock.settimeout(self.connection_timeout / 1000)
        self.sock.connect((self._ip, self._port))
        self.info("Connected to server successfully")
        Thread(target=self.__receive).start()

    def __receive(self):
        while True:
            try:
                data = self.sock.recv(70656).decode('utf-8')
                if not data:
                    self.error("Connection closed by server")
                    break
                self.info(f"Received data: {data}")
                self.message_buffer.append(json.loads(data))
            except Exception as e:
                self.error(f"An error occurred while receiving data: {e}")
                break

    def send(self, data: dict):
        try:
            self.sock.sendall(json.dumps(data).encode('utf-8'))
        except Exception as e:
            self.error(f"Failed to send data: {e}")
