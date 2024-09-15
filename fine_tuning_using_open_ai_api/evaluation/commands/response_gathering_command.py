import json
import os
from time import sleep

from hugchat import hugchat
from hugchat.login import Login
from openai import OpenAI
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ResponseGathering(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

        self.hugging_chat_password = None
        self.hugging_chat_email = None
        self.hugging_chatbot = None
        self.hugging_models = []
        self.client = OpenAI()
        self.base_model = "gpt-4o-mini-2024-07-18"
        self.base_model_default_content = """
                You are a general chat-bot and You must avoid all kind of professional opinions on sensitive matters such as health or other specialized fields. For such matters, consulting a professional is always recommended.
        """
        self.fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal::A093zQF2"
        self.test_datasets = {
            "test": [],
            "unseen_test": [],
        }
        self.prompt = 'Answer only in this format and dont say anything more (Only common name of the most probable disease in a JSON Object as value): "{"disease_name" : "{predicted_disease_name}"}". '
        self.responses = {}

    def setup(self):
        with open("fine_tuning_using_open_ai_api/data/qa_dataset_test.jsonl", 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                self.test_datasets["test"].append(data)

        with open("fine_tuning_using_open_ai_api/data/qa_dataset_unseen_test.jsonl", 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                self.test_datasets["unseen_test"].append(data)

        for key in self.test_datasets:
            self.responses[key] = {
                "reference": [],
                "base_model": [],
                "fine_tuned_model": [],
                "meta-llama/Meta-Llama-3.1-70B-Instruct": [],
                "CohereForAI/c4ai-command-r-plus-08-2024": [],
                "mistralai/Mixtral-8x7B-Instruct-v0.1": [],
                "microsoft/Phi-3-mini-4k-instruct": [],
            }

        self.hugging_chat_email = os.getenv("HUGGING_CHAT_EMAIL")
        self.hugging_chat_password = os.getenv("HUGGING_CHAT_PASSWORD")
        sign = Login(self.hugging_chat_email, self.hugging_chat_password)
        cookies = sign.login(cookie_dir_path="./cookies", save_cookies=True)
        self.hugging_chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        hugging_models = self.hugging_chatbot.get_available_llm_models()
        for model in hugging_models:
            self.hugging_models.append(model.displayName)

    def get_gpt_response(self, model, prompt, system_content):
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

    def get_model_index(self, model_name):
        return self.hugging_models.index(model_name)

    def get_hugging_chat_response(self, model_name, prompt, system_content):
        self.hugging_chatbot.switch_llm(self.get_model_index(model_name))
        self.hugging_chatbot.new_conversation(switch_to=True)
        response = self.hugging_chatbot.chat(text=self.prompt + "\n" + prompt, web_search=False)
        return response.text

    def execute(self):
        for key in self.test_datasets:
            for index, data in enumerate(self.test_datasets[key]):
                system_content = data["messages"][0]["content"]
                prompt = data["messages"][1]["content"]
                reference = data["messages"][2]["content"]

                for model_name in self.responses[key]:
                    if model_name == "reference":
                        self.responses[key][model_name].append(reference)
                    elif model_name == "base_model":
                        self.responses[key][model_name].append(
                            self.get_gpt_response(self.base_model, prompt, self.base_model_default_content)
                        )
                    elif model_name == "fine_tuned_model":
                        self.responses[key][model_name].append(
                            self.get_gpt_response(self.fine_tuned_model, prompt, system_content)
                        )
                    else:
                        sleep(15)
                        self.responses[key][model_name].append(
                            self.get_hugging_chat_response(model_name, prompt, system_content)
                        )
                print(
                    f"Completed for {index + 1} out of {len(self.test_datasets[key])} of {len(self.test_datasets)} dataset"
                )
                self.cleanup()

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/responses.json", "w") as out:
            json.dump(self.responses, out, indent=4)
