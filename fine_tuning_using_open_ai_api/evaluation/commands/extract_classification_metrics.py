import json
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ExtractingClassificationMetricsCommand(BaseCommand):
    def __init__(self):
        self.models = {
            "base_model": {},
            "fine_tuned_model": {},
            "meta-llama/Meta-Llama-3.1-70B-Instruct": {},
            "CohereForAI/c4ai-command-r-plus-08-2024": {},
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {},
            "microsoft/Phi-3-mini-4k-instruct": {},
        }
        self.responses = {}

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/responses.json') as f:
            self.responses = json.load(f)
            for key in self.responses:
                for index, model_name in enumerate(self.responses[key]):
                    for i, data_row in enumerate(self.responses[key][model_name]):
                        try:
                            self.responses[key][model_name][i] = json.loads(self.responses[key][model_name][i])[
                                'disease_name']
                        except:
                            self.responses[key][model_name][i] = ""

        for key in self.responses:
            for model_name in self.models:
                self.models[model_name][key] = {
                    "accuracy": 0,
                    "error_rate": 0,
                    # "parsing_error_rate": 0,
                }

    def string_cleaner(self, string):
        string = string.lower().strip()
        for char in ["_", "."]:
            string = string.replace(char, '').strip()
        string = string.replace(' ', '_').strip('_')
        return string

    def is_similar(self, str1, str2, threshold=0.9):
        if str1 == "" or str2 == "":
            return False
        str2_split = str2.split("_")
        str1_split = str1.split("_")
        for str1_item in str1_split:
            for str2_item in str2_split:
                if str1_item == str2_item:
                    return True
        return str1.__contains__(str2) or str2.__contains__(str1)

    def execute(self):
        for key in self.responses:
            reference_responses = [self.string_cleaner(item) for item in self.responses[key]["reference"]]
            for model_name in self.models:
                model_responses = [self.string_cleaner(item) for item in self.responses[key][model_name]]

                correct_predictions = 0
                parsing_error_predictions = 0
                total_predictions = 0

                for index, (ref, model) in enumerate(zip(reference_responses, model_responses)):
                    if model == "":
                        parsing_error_predictions += 1
                    elif self.is_similar(ref, model):
                        correct_predictions += 1
                    total_predictions += 1

                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                parsing_error_rate = parsing_error_predictions / total_predictions if total_predictions > 0 else 0
                error_rate = 1 - accuracy
                self.models[model_name][key]["accuracy"] = accuracy
                self.models[model_name][key]["error_rate"] = error_rate
                # self.models[model_name][key]["parsing_error_rate"] = parsing_error_rate

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/scores/results.json", "w") as out:
            json.dump(self.models, out, indent=4)
