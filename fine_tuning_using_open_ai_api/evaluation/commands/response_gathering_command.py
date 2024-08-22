import json
from openai import OpenAI
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ResponseGathering(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

        self.client = OpenAI()
        self.base_model = "gpt-4o-mini-2024-07-18"
        self.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal::9uW8lVcz"
        self.test_dataset = []
        self.prompt = "Answer only in this format : 'The disease associated with these symptoms could be {disease}' and dont say anything more about that. "
        self.responses = {
            "reference": [],
            "base_model": [],
            "fine_tuned_model": [],
        }

    def setup(self):
        with open("fine_tuning_using_open_ai_api/data/qa_dataset_test.jsonl", 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                self.test_dataset.append(data)

    def get_response(self, model, prompt, system_content):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": self.prompt + prompt
                }
            ]

        )
        return response.choices[0].message.content.strip().lower()

    def execute(self):
        for index, data in enumerate(self.test_dataset):
            system_content = data["messages"][0]["content"]
            prompt = data["messages"][1]["content"]
            reference = data["messages"][2]["content"]

            base_model_response = self.get_response(self.base_model, prompt, system_content)
            fine_tuned_model_response = self.get_response(self.fine_tuned_model, prompt, system_content)

            self.responses["reference"].append(reference)
            self.responses["base_model"].append(base_model_response)
            self.responses["fine_tuned_model"].append(fine_tuned_model_response)

            print(f"Completed for {index + 1} out of {len(self.test_dataset)}")

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/responses.json", "w") as out:
            json.dump(self.responses, out, indent=4)
