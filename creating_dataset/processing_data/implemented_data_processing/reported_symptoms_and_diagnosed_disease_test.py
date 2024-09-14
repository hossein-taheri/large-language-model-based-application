import json
from creating_dataset.processing_data.data_processing_base import DataProcessingBase


class ReportedSymptomsAndDiagnosedDiseaseTest(DataProcessingBase):
    def __init__(self):
        super().__init__(dataset_name="reported_symptoms_and_diagnosed_disease_test")

    def prepare_data(self):
        print("Preparing data on dataset ::", self.dataset_name)

        file = open(f"creating_dataset/data/unprocessed_data/{self.dataset_name}/raw_dataset.txt").read()

        lines = file.strip().split('\n')[2:]

        keys = [
            "Age",
            "Gender",
            "Chief Complaint (CC)",
            "Present Illness (PI)",
            "Reported Symptoms",
            "Prescribed Medication",
            "Diagnosed Disease"
        ]
        data = []

        for line in lines:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            record = dict(zip(keys, cells))
            data.append(record)

        self.qa_json = data

    def generate_qa_dataset(self):
        print("Generating data on dataset ::", self.dataset_name)

        qa_dataset = []

        for disease in self.qa_json:
            qa_dataset.append(
                self.generate_qa_pairs(disease)
            )

        self.qa_json = qa_dataset

    @staticmethod
    def generate_qa_pairs(disease):
        qa_pairs = {
            'prompt': f"What disease is associated with these symptoms: {disease['Reported Symptoms'].lower()}?",
            'completion': f'{{"disease_name" : "{disease["Diagnosed Disease"].lower()}"}}'
        }

        return qa_pairs
