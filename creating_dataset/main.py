from creating_dataset.processing_data.QADatasetManagement import QADatasetManagement
from creating_dataset.processing_data.implemented_data_processing.DiseasePredictionUsingMachineLearning import \
    DiseasePredictionUsingMachineLearning


def generate_datasets():
    qa_management = QADatasetManagement(qa_directory='creating_dataset/data/qa_raw_data/')
    qa_management.clear_dataset_directory()

    implemented_data_processing_classes = [
        DiseasePredictionUsingMachineLearning,
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
