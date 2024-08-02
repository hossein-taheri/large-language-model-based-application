import pandas as pd

from creating_dataset.processing_data.DataProcessingBase import DataProcessingBase


class DiseasePredictionUsingMachineLearning(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="disease_prediction_using_machine_learning")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        training_df = pd.read_csv(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/Training.csv")
        training_df.drop("Unnamed: 133", axis=1, inplace=True)
        testing_df = pd.read_csv(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/Testing.csv")
        combined_df = pd.concat([training_df, testing_df])

        self.dataframe = combined_df

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)

        qa_dataset = []
        symptom_columns = self.dataframe.columns[:-1]

        for _, row in self.dataframe.iterrows():
            qa_dataset.append(self.generate_qa_pairs(symptom_columns, row))

        self.qa_json = qa_dataset

    @staticmethod
    def generate_qa_pairs(symptom_columns, row):
        symptoms = ', '.join([symptom.replace('_', ' ') for symptom in symptom_columns if row[symptom] == 1])
        disease = row['prognosis']

        qa_pairs = [
            {
                'prompt': f"What disease is associated with these symptoms: {symptoms}?",
                'completion': f"The disease associated with these symptoms is {disease}."
            }
        ]

        return qa_pairs
