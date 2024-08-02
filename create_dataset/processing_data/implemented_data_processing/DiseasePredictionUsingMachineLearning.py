import pandas as pd

from create_dataset.processing_data.DataProcessingBase import DataProcessingBase


class DiseasePredictionUsingMachineLearning(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="disease_prediction_using_machine_learning")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        df = pd.read_csv(f"create_dataset/data/unprocessed_data/{self.dataset_name}/Training.csv")
        print(df.head(20))

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)
        pass
