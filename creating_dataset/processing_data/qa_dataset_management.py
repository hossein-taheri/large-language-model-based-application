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
        data = []
        test_data = []
        for file in os.listdir(self.qa_directory):
            if file.endswith('.json'):
                json_file = os.path.join(self.qa_directory, file)
                f = open(json_file, 'r')
                if not file.__contains__('test'):
                    data += json.loads(f.read())
                else:
                    test_data += json.loads(f.read())

        with open(os.path.join(self.qa_directory, "json_qa_dataset.jsonl"), 'w') as file:
            for item in data:
                json.dump(item, file)
                file.write('\n')

        with open(os.path.join(self.qa_directory, "json_qa_dataset_test.jsonl"), 'w') as file:
            for item in test_data:
                json.dump(item, file)
                file.write('\n')

        all_data = [data, test_data]
        for all_data_index in range(len(all_data)):
            new_data = []
            for prompt_answer_pair in all_data[all_data_index]:
                new_data.append({
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
            all_data[all_data_index] = new_data
        data, test_data = all_data

        with open(os.path.join(self.qa_directory, "qa_dataset.jsonl"), 'w') as file:
            for item in data:
                json.dump(item, file)
                file.write('\n')

        with open(os.path.join(self.qa_directory, "qa_dataset_test.jsonl"), 'w') as file:
            for item in test_data:
                json.dump(item, file)
                file.write('\n')
