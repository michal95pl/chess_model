import json
import logging

class ConfigFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.logger = logging.getLogger('app_logger')

        try :
            with open(filepath, 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            self.logger.error("Config file not found!")
            raise FileNotFoundError("Config file not found!")

    def get_config(self, key):
        return self.config.get(key, None)

    def get_config2(self, key1, key2):
        return self.config[key1][key2]