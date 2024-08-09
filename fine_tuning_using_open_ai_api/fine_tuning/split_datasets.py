import random


def split_jsonl(file_paths, train_ratio=0.95):
    with open(file_paths['main_dataset'], 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    split_index = int(len(lines) * train_ratio)

    train_lines = lines[:split_index]
    val_lines = lines[split_index:]

    with open(file_paths['train_dataset'], 'w') as train_file:
        train_file.writelines(train_lines)

    with open(file_paths['validation_dataset'], 'w') as val_file:
        val_file.writelines(val_lines)

    print(f"Data split into {len(train_lines)} training and {len(val_lines)} validation lines.")


def split_dataset_into_train_and_test_datasets():
    split_jsonl(
        {
            'main_dataset': 'fine_tuning_using_open_ai_api/data/qa_dataset.jsonl',
            'train_dataset': 'fine_tuning_using_open_ai_api/data/qa_dataset_train.jsonl',
            'validation_dataset': 'fine_tuning_using_open_ai_api/data/qa_dataset_val.jsonl',
        }
    )