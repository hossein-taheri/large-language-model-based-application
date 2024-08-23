import json

from creating_dataset.commands.base_command import BaseCommand


class ChangeDatasetsFormat(BaseCommand):

    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.qa_directory = 'creating_dataset/data/qa_raw_data/'

    def setup(self):
        pass

    def execute(self):
        def change_datasets_format(data_paths):
            for paths in data_paths:
                with open(paths[0], 'r') as file:
                    lines = file.readlines()
                user_assistant_dataset = []
                for prompt_answer_pair in lines:
                    prompt_answer_pair = json.loads(prompt_answer_pair)
                    user_assistant_dataset.append({
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an empathetic medical assistant to help detection of disease.",
                            },
                            {
                                "role": "user",
                                "content": prompt_answer_pair['prompt'],
                            },
                            {
                                "role": "assistant",
                                "content": prompt_answer_pair['completion'],
                            },
                        ]
                    })

                with open(paths[1], 'w') as file:
                    for item in user_assistant_dataset:
                        json.dump(item, file)
                        file.write('\n')

        change_datasets_format(
            data_paths=[
                (
                    "creating_dataset/data/qa_raw_data/json_qa_dataset.jsonl",
                    "creating_dataset/data/qa_raw_data/qa_dataset.jsonl"
                ),
                (
                    "creating_dataset/data/qa_raw_data/json_qa_dataset_unseen_test.jsonl",
                    "creating_dataset/data/qa_raw_data/qa_dataset_unseen_test.jsonl"
                )
            ]
        )

    def cleanup(self):
        pass
