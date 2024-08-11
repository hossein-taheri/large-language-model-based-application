import json
import os


class QADatasetManagement:
    def __init__(self, qa_directory):
        self.qa_directory = qa_directory

    def clear_dataset_directory(self):
        for file in os.listdir(self.qa_directory):
            if file.endswith('.json') or file.endswith('.jsonl'):
                json_file = os.path.join(self.qa_directory, file)
                os.remove(json_file)

    def generate_one_dataset_file(self):
        json_data = []
        for file in os.listdir(self.qa_directory):
            if file.endswith('.json'):
                json_file = os.path.join(self.qa_directory, file)
                with open(json_file, 'r') as f:
                    json_data += json.loads(f.read())

        with open(os.path.join(self.qa_directory, "qa_dataset.jsonl"), 'w') as file:
            for item in json_data:
                json.dump(item, file)
                file.write('\n')
