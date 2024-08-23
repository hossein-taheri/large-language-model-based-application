import random

from creating_dataset.commands.base_command import BaseCommand


class SplitDatasets(BaseCommand):

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.qa_directory = 'creating_dataset/data/qa_raw_data/'

    def setup(self):
        pass

    def execute(self):
        def split_jsonl(file_paths, train_ratio=0.90, val_ratio=0.05):
            with open(file_paths['main_dataset'], 'r') as file:
                lines = file.readlines()

            random.shuffle(lines)

            train_split_index = int(len(lines) * train_ratio)
            val_split_index = int(len(lines) * (train_ratio + val_ratio))
            train_lines = lines[:train_split_index]
            val_lines = lines[train_split_index:val_split_index]
            test_lines = lines[val_split_index:]

            with open(file_paths['train_dataset'], 'w') as train_file:
                train_file.writelines(train_lines)

            with open(file_paths['validation_dataset'], 'w') as val_file:
                val_file.writelines(val_lines)

            with open(file_paths['test_dataset'], 'w') as val_file:
                val_file.writelines(test_lines)

            print(f"Data split into {len(train_lines)} training and {len(val_lines)} validation lines.")

        split_jsonl(
            {
                'main_dataset': 'creating_dataset/data/qa_raw_data/qa_dataset.jsonl',
                'train_dataset': 'creating_dataset/data/qa_raw_data/qa_dataset_train.jsonl',
                'validation_dataset': 'creating_dataset/data/qa_raw_data/qa_dataset_val.jsonl',
                'test_dataset': 'creating_dataset/data/qa_raw_data/qa_dataset_test.jsonl',
            }
        )

    def cleanup(self):
        pass
