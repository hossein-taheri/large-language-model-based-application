from fine_tuning_using_open_ai_api.fine_tuning.data_immigration import immigrate_data
from fine_tuning_using_open_ai_api.fine_tuning.price_estimation import predict_qa_dataset_price


def main():
    immigrate_data()
    predict_qa_dataset_price()


if __name__ == '__main__':
    main()
