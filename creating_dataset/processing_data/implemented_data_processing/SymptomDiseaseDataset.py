import json

import pandas as pd

from creating_dataset.processing_data.DataProcessingBase import DataProcessingBase


class SymptomDiseaseDataset(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="symptom_disease_dataset_duxprajapati")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        training_df = pd.read_csv(
            f"creating_dataset/data/unprocessed_data/{self.dataset_name}/symptom-disease-test-dataset.csv")
        testing_df = pd.read_csv(
            f"creating_dataset/data/unprocessed_data/{self.dataset_name}/symptom-disease-train-dataset.csv")
        combined_df = pd.concat([training_df, testing_df])
        label_mapping = {
            y: x for x, y in json.loads(
                open(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/mapping.json").read()
            ).items()
        }
        combined_df['label'] = combined_df['label'].map(label_mapping)
        self.dataframe = combined_df

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)

        qa_dataset = []

        for _, row in self.dataframe.iterrows():
            qa_dataset.append(self.generate_qa_pairs(row))

        self.qa_json = qa_dataset

    @staticmethod
    def generate_qa_pairs(row):
        qa_pairs = [
            {
                'prompt': row['text'],
                'completion': row['label']
            }
        ]

        return qa_pairs
