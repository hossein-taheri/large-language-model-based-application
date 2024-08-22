import json
import tiktoken
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class PriceEstimationCommand(BaseCommand):
    def __init__(self, file_path="fine_tuning_using_open_ai_api/data/qa_dataset.jsonl"):
        super().__init__(name=self.__class__.__name__)
        self.encodings = {
            'gpt-3.5-turbo': tiktoken.encoding_for_model("gpt-3.5-turbo"),
            'gpt-4o-mini-2024-07-18': tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18"),
        }
        self.cost_per_1m_tokens = {
            'gpt-3.5-turbo': 8,
            'gpt-4o-mini-2024-07-18': 3,
        }
        self.jsonl_file = None
        self.jsonl_file_path = file_path

    def setup(self):
        with open(self.jsonl_file_path, 'r') as f:
            self.jsonl_file = [json.loads(line) for line in f]

    def execute(self):
        total_tokens_per_model = {
            model_name: sum(
                sum(
                    len(
                        self.encodings[model_name].encode(example['messages'][index]['content'])
                    ) for index in range(len(example['messages']))
                ) for example in self.jsonl_file
            ) for model_name in self.encodings
        }

        fine_tuning_costs = {
            model: (total_tokens / 1_000_000) * self.cost_per_1m_tokens[model] for model, total_tokens in
            total_tokens_per_model.items()
        }

        for model, total_tokens in total_tokens_per_model.items():
            print(f"The estimated total tokens of your data in {model} is {total_tokens}")

        print("\n")

        for model, cost in fine_tuning_costs.items():
            print(f"The estimated cost of fine-tuning {model} is ${cost:.2f}")

        print("\n")

    def cleanup(self):
        pass
