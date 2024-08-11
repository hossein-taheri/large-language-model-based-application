import pandas as pd
from creating_dataset.processing_data.data_processing_base import DataProcessingBase


class MedicalQuestionAnsweringDataset(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="medical_question_answering_dataset")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        df = pd.read_csv(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/medquad.csv")
        df.drop(['source', 'focus_area'], axis=1, inplace=True)
        self.dataframe = df

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)

        qa_dataset = []

        for _, row in self.dataframe.iterrows():
            qa_dataset.append({
                "prompt": row["question"],
                "completion": row["answer"],
            })

        self.qa_json = qa_dataset
