from utils.logger import Logger
import socket
from queue import Queue
from threading import Thread
import json

class Communication(Thread):
    def __init__(self, port: int, ip: str = 'localhost'):
        super().__init__(daemon=True, name="CommunicationThread")
        self. receive_thread_running = True
        self.logger = Logger()
        self.queue = Queue(10)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ip, port))
        self.logger.info(f"Socket created on {ip}:{port}")
        self.socket.listen()

        self.start()

    def run(self):
        while self.receive_thread_running:
            try:
                conn, addr = self.socket.accept()
                self.logger.info(f"Connection accepted from {addr}")
                Thread(target=self.__receive, args=(conn, addr, ), daemon=True).start()
            except OSError as e:
                if self.receive_thread_running:
                    self.logger.error(f"Socket error: {e}")
                break

    def __receive(self, conn, addr):
        with conn:
            while self.receive_thread_running:
                try:
                    data = conn.recv(70656)
                    if not data:
                        break
                    try:
                        message = json.loads(data.decode('utf-8'))
                        self.queue.put((conn, message))
                        self.logger.debug(f"Received data from {addr}: {message}")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error: {e}")
                except OSError as e:
                    self.logger.error(f"Socket error: {e}")
                    break

    def send(self, data: dict, conn):
        addr = conn.getpeername()
        try:
            message = json.dumps(data).encode('utf-8')
            conn.send(message)
            self.logger.debug(f"Sent data to {addr}: {message}")
        except Exception as e:
            self.logger.error(f"Error sending data to {addr}: {e}")

    def get_first_message(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def is_message_available(self):
        return not self.queue.empty()

    def close(self):
        self.receive_thread_running = False
        self.socket.shutdown(socket.SHUT_RDWR) # disable further sends and receives to unblock accept and rcv in threads
        self.socket.close()

        self.logger.info("Socket closed")
