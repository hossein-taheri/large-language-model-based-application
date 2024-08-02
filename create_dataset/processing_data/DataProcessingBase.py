import abc


class DataProcessingBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataframe = None

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def generate_qa_dataset(self):
        pass
