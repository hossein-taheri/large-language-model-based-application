from create_dataset.processing_data.implemented_data_processing.DiseasePredictionUsingMachineLearning import \
    DiseasePredictionUsingMachineLearning


def generate_dataset():
    implemented_data_processing_classes = [
        DiseasePredictionUsingMachineLearning,
    ]
    for implemented_data_processing_class in implemented_data_processing_classes:
        implemented_data_processing = implemented_data_processing_class()
        implemented_data_processing.prepare_data()
        implemented_data_processing.generate_qa_dataset()
        implemented_data_processing.save_results()


def main():
    generate_dataset()


if __name__ == '__main__':
    main()
