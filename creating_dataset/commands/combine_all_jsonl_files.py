import os
import json

from creating_dataset.commands.base_command import BaseCommand


class CombineAllJsonlFiles(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.qa_directory = 'creating_dataset/data/qa_raw_data/'
        self.data = []
        self.test_data = []

    def setup(self):
        pass

    def execute(self):
        for file in os.listdir(self.qa_directory):
            if file.endswith('.json'):
                json_file = os.path.join(self.qa_directory, file)
                f = open(json_file, 'r')
                if not file.__contains__('test'):
                    self.data += json.loads(f.read())
                else:
                    self.test_data += json.loads(f.read())

    def cleanup(self):
        with open(os.path.join(self.qa_directory, "json_qa_dataset.jsonl"), 'w') as file:
            for item in self.data:
                json.dump(item, file)
                file.write('\n')

        with open(os.path.join(self.qa_directory, "json_qa_dataset_unseen_test.jsonl"), 'w') as file:
            for item in self.test_data:
                json.dump(item, file)
                file.write('\n')
