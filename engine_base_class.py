from abc import ABC, abstractmethod

class EngineBaseClass(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self):
        return

    @abstractmethod
    def load_model(self, model):
        return
        
    @abstractmethod
    def parse(self, sentence):
        return sentence
