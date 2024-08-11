import json
from creating_dataset.processing_data.data_processing_base import DataProcessingBase


class DiseaseSymptomKnowledgeDatabase(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="disease_symptom_knowledge_database")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        json_formated_dataset = json.loads(
            open(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/dataset.json").read()
        )

        self.dataframe = json_formated_dataset

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)

        qa_dataset = []

        for disease in self.dataframe:
            qa_dataset.append(
                self.generate_qa_pairs(disease, self.dataframe[disease])
            )

        self.qa_json = qa_dataset

    @staticmethod
    def generate_qa_pairs(disease, symptoms):
        symptoms = ', '.join([symptom.replace('_', ' ') for symptom in symptoms])
        disease = disease.replace('_', ' ')

        qa_pairs = {
            'prompt': f"What disease is associated with these symptoms: {symptoms}?",
            'completion': f"The disease associated with these symptoms is {disease}."
        }

        return qa_pairs
