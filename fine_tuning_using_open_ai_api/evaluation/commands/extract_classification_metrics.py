import json
from difflib import SequenceMatcher
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ExtractingClassificationMetricsCommand(BaseCommand):
    def __init__(self):
        self.base_model_scores = {}
        self.fine_tuned_model_scores = {}
        self.responses = {}

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/responses.json') as f:
            self.responses = json.load(f)
            for key in self.responses:
                for index, model_response in enumerate(self.responses[key]):
                    for i, data_row in enumerate(self.responses[key][model_response]):
                        print(i, data_row, model_response, key)
                        self.responses[key][model_response][i] = json.loads(self.responses[key][model_response][i])[
                            'disease_name']

        for key in self.responses:
            self.base_model_scores[key] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
            self.fine_tuned_model_scores[key] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    def string_cleaner(self, string):
        string = string.lower().strip()
        for char in ["_", "."]:
            string = string.replace(char, '').strip()
        string = string.replace(' ', '_').strip('_')
        return string

    def is_similar(self, str1, str2, threshold=0.9):
        str2_split = str2.split("_")
        str1_split = str1.split("_")
        # print(str1_split, str2_split)
        for str1_item in str1_split:
            for str2_item in str2_split:
                if str1_item == str2_item:
                    return True
        # similarity_ratio = SequenceMatcher(None, str1, str2).ratio()
        return str1.__contains__(str2) or str2.__contains__(str1)

    def execute(self):
        for key in self.responses:
            reference_responses = [self.string_cleaner(item) for item in self.responses[key]["reference"]]
            base_model_responses = [self.string_cleaner(item) for item in self.responses[key]["base_model"]]
            fine_tuned_model_responses = [self.string_cleaner(item) for item in self.responses[key]["fine_tuned_model"]]

            adjusted_base_model_responses = []
            adjusted_fine_tuned_model_responses = []
            for index, (ref, model) in enumerate(zip(reference_responses, base_model_responses)):
                if self.is_similar(ref, model):
                    adjusted_base_model_responses.append(ref)
                else:
                    adjusted_base_model_responses.append(model)
            for index, (ref, model) in enumerate(zip(reference_responses, fine_tuned_model_responses)):
                if self.is_similar(ref, model):
                    adjusted_fine_tuned_model_responses.append(ref)
                else:
                    if key == "unseen_test":
                        print(index, model)
                    adjusted_fine_tuned_model_responses.append(model)

            base_accuracy = accuracy_score(
                reference_responses,
                adjusted_base_model_responses
            )
            base_precision = precision_score(
                reference_responses,
                adjusted_base_model_responses,
                average='macro',
                zero_division=1
            )
            base_recall = recall_score(
                reference_responses,
                adjusted_base_model_responses,
                average='macro',
                zero_division=1
            )
            base_f1 = f1_score(
                reference_responses,
                adjusted_base_model_responses,
                average='macro',
                zero_division=1
            )

            self.base_model_scores[key]["accuracy"] = base_accuracy
            self.base_model_scores[key]["precision"] = base_precision
            self.base_model_scores[key]["recall"] = base_recall
            self.base_model_scores[key]["f1"] = base_f1

            fine_tuned_accuracy = accuracy_score(
                reference_responses,
                adjusted_fine_tuned_model_responses
            )
            fine_tuned_precision = precision_score(
                reference_responses,
                adjusted_fine_tuned_model_responses,
                average='macro',
                zero_division=1
            )
            fine_tuned_recall = recall_score(
                reference_responses,
                adjusted_fine_tuned_model_responses,
                average='macro',
                zero_division=1
            )
            fine_tuned_f1 = f1_score(
                reference_responses,
                adjusted_fine_tuned_model_responses,
                average='macro',
                zero_division=1
            )

            self.fine_tuned_model_scores[key]["accuracy"] = fine_tuned_accuracy
            self.fine_tuned_model_scores[key]["precision"] = fine_tuned_precision
            self.fine_tuned_model_scores[key]["recall"] = fine_tuned_recall
            self.fine_tuned_model_scores[key]["f1"] = fine_tuned_f1

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/scores/base_model_metric_scores.json", "w") as out:
            json.dump(self.base_model_scores, out, indent=4)

        with open("fine_tuning_using_open_ai_api/data/results/scores/fine_tuned_model_metric_scores.json", "w") as out:
            json.dump(self.fine_tuned_model_scores, out, indent=4)
