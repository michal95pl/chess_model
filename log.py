import time


class Log:

    def __init__(self, path: str = None):
        if path is not None:
            self.file = open(path, "a")

    def info(self, message: str):
        print("[" + time.strftime("%H:%M:%S") + "] [INFO] " + message)
        if self.file is not None:
            self.file.write("[INFO] [" + time.strftime("%H:%M:%S") + "] " + message + "\n")
            self.file.flush()

    def error(self, message: str):
        print("[" + time.strftime("%H:%M:%S") + "] [ERROR] " + message)
        if self.file is not None:
            self.file.write("[ERROR] [" + time.strftime("%H:%M:%S") + "] " + message + "\n")
            self.file.flush()
