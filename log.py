import time


class Log:

    file = open("log.txt", "a")
    @staticmethod
    def info(message: str):
        print("[INFO] [" + time.strftime("%H:%M:%S") + "] " + message)
        Log.file.write("[INFO] [" + time.strftime("%H:%M:%S") + "] " + message + "\n")
        Log.file.flush()

    @staticmethod
    def error(message: str):
        print("[ERROR] [" + time.strftime("%H:%M:%S") + "] " + message)
        Log.file.write("[ERROR] [" + time.strftime("%H:%M:%S") + "] " + message + "\n")
        Log.file.flush()
