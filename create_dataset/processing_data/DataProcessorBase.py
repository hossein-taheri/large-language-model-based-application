import abc


class DataProcessorBase(abc.ABC):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def prepare_data(self):
        pass

    def generate_qa_dataset(self):
        pass
