from commands_handler import CommandsHandler
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    CommandsHandler().run_app()