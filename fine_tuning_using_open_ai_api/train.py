from fine_tuning_using_open_ai_api.fine_tuning.train_model import fine_tune_model
from fine_tuning_using_open_ai_api.utils import import_env_variables

def main():
    train_dataset_path = 'fine_tuning_using_open_ai_api/data/qa_dataset_train.jsonl'
    validation_dataset_path = 'fine_tuning_using_open_ai_api/data/qa_dataset_val.jsonl'

    fine_tune_model(
        train_dataset_path,
        validation_dataset_path
    )


if __name__ == '__main__':
    main()
