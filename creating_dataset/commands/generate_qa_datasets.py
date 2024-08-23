import os

from creating_dataset.commands.base_command import BaseCommand
from creating_dataset.processing_data.implemented_data_processing.disease_prediction_using_machine_learning import \
    DiseasePredictionUsingMachineLearning
from creating_dataset.processing_data.implemented_data_processing.medical_question_answering_dataset import \
    MedicalQuestionAnsweringDataset
from creating_dataset.processing_data.implemented_data_processing.disease_symptom_knowledge_database import \
    DiseaseSymptomKnowledgeDatabase
from creating_dataset.processing_data.implemented_data_processing.symptom_disease_dataset import SymptomDiseaseDataset
from creating_dataset.processing_data.implemented_data_processing.reported_symptoms_and_diagnosed_disease_test import \
    ReportedSymptomsAndDiagnosedDiseaseTest


class GenerateQADatasets(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.qa_directory = 'creating_dataset/data/qa_raw_data/'

    def setup(self):
        pass

    def execute(self):
        self.clear_dataset_directory()

        implemented_data_processing_classes = [
            DiseasePredictionUsingMachineLearning,
            # MedicalQuestionAnsweringDataset, # This is a big dataset
            # DiseaseSymptomKnowledgeDatabase, # Not a useful dataset
            SymptomDiseaseDataset,
            ReportedSymptomsAndDiagnosedDiseaseTest,
        ]
        for implemented_data_processing_class in implemented_data_processing_classes:
            implemented_data_processing = implemented_data_processing_class()
            implemented_data_processing.prepare_data()
            implemented_data_processing.generate_qa_dataset()
            implemented_data_processing.save_results()

    def clear_dataset_directory(self):
        for file in os.listdir(self.qa_directory):
            if file.endswith('.json') or file.endswith('.jsonl'):
                json_file = os.path.join(self.qa_directory, file)
                os.remove(json_file)

    def cleanup(self):
        pass
