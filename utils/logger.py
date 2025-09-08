import time

class Logger:

    log_level = "DEBUG"

    def __init__(self):
        self.log_file = open('chess_model.log', 'a')

    def info(self, message: str):
        self.log_file.write(f"[{time.strftime('%H:%M:%S')}] [INFO] {message}\n")
        self.log_file.flush()
        print(f"[{time.strftime('%H:%M:%S')}] [INFO] {message}")

    def error(self, message: str):
        self.log_file.write(f"[{time.strftime('%H:%M:%S')}] [ERROR] {message}\n")
        self.log_file.flush()
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {message}")

    def debug(self, message: str):
        if Logger.log_level == "DEBUG":
            self.log_file.write(f"[{time.strftime('%H:%M:%S')}] [DEBUG] {message}\n")
            self.log_file.flush()
            print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] {message}")

    def warning(self, message: str):
        self.log_file.write(f"[{time.strftime('%H:%M:%S')}] [WARNING] {message}\n")
        self.log_file.flush()
        print(f"[{time.strftime('%H:%M:%S')}] [WARNING] {message}")