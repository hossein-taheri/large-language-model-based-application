from create_dataset.processing_data.implemented_data_processing.DiseasePredictionUsingMachineLearning import \
    DiseasePredictionUsingMachineLearning


def process_data():
    implemented_data_processing_classes = [
        DiseasePredictionUsingMachineLearning(),
    ]
    for implemented_data_processing_class in implemented_data_processing_classes:
        implemented_data_processing_class.prepare_data()
        implemented_data_processing_class.generate_qa_dataset()


def main():
    process_data()


if __name__ == '__main__':
    main()
