import traceback
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def cleanup(self):
        pass

    def run(self):
        try:
            self.setup()
            self.execute()
            self.cleanup()
        except Exception as e:
            print(f"Error during execution: {e.__str__()}")
            print(traceback.format_exc())
