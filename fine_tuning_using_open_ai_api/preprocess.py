from fine_tuning_using_open_ai_api.fine_tuning.data_immigration import immigrate_data
from fine_tuning_using_open_ai_api.fine_tuning.price_estimation import predict_qa_dataset_price
from fine_tuning_using_open_ai_api.fine_tuning.split_datasets import split_dataset_into_train_and_test_datasets
from utils import import_env_variables


def main():
    source_dataset_path = 'creating_dataset/data/qa_raw_data/qa_dataset.jsonl'
    dataset_path = 'fine_tuning_using_open_ai_api/data/qa_dataset.jsonl'
    # immigrate_data(source_dataset_path, dataset_path)
    # predict_qa_dataset_price(dataset_path)
    # split_dataset_into_train_and_test_datasets()


if __name__ == '__main__':
    main()
