import abc
import json


class DataProcessingBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataframe = None
        self.qa_json = None

    @abc.abstractmethod
    def prepare_data(self):
        pass

    @abc.abstractmethod
    def generate_qa_dataset(self):
        pass

    def save_results(self):
        with open(f'creating_dataset/data/qa_raw_data/{self.dataset_name}.json', 'wb') as f:
            f.write(json.dumps(self.qa_json, indent=4).encode())
