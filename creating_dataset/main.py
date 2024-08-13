from creating_dataset.processing_data.data_processing_base import DataProcessingBase
from creating_dataset.processing_data.qa_dataset_management import QADatasetManagement
from creating_dataset.processing_data.implemented_data_processing.disease_prediction_using_machine_learning import \
    DiseasePredictionUsingMachineLearning
from creating_dataset.processing_data.implemented_data_processing.medical_question_answering_dataset import \
    MedicalQuestionAnsweringDataset
from creating_dataset.processing_data.implemented_data_processing.disease_symptom_knowledge_database import \
    DiseaseSymptomKnowledgeDatabase
from creating_dataset.processing_data.implemented_data_processing.symptom_disease_dataset import SymptomDiseaseDataset
from creating_dataset.processing_data.implemented_data_processing.reported_symptoms_and_diagnosed_disease_test import \
    ReportedSymptomsAndDiagnosedDiseaseTest


def generate_datasets():
    qa_management = QADatasetManagement(qa_directory='creating_dataset/data/qa_raw_data/')
    qa_management.clear_dataset_directory()

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

    qa_management.generate_one_dataset_file()


def main():
    generate_datasets()


if __name__ == '__main__':
    main()
