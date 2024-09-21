import json

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ExtractingMetricsCommand(BaseCommand):
    def __init__(self):
        self.responses = {}
        self.models = {
            "base_model": {},
            "fine_tuned_model": {},
            "meta-llama/Meta-Llama-3.1-70B-Instruct": {},
            "CohereForAI/c4ai-command-r-plus-08-2024": {},
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {},
            "microsoft/Phi-3-mini-4k-instruct": {},
        }

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/responses.json') as f:
            self.responses = json.load(f)
        for key in self.responses:
            for model_name in self.models:
                self.models[model_name][key] = {
                    "bleu": [],
                    "rouge1": [],
                    "rouge2": [],
                    "rougeL": []
                }

    def calculate_bleu(self, reference, candidate):
        reference = [reference.split()]
        candidate = candidate.split()
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
        return bleu_score

    def calculate_rouge(self, reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores

    def execute(self):
        for key in self.responses:
            for index, response in enumerate(self.responses[key]["reference"]):
                reference = self.responses[key]["reference"][index]
                for model_name in self.models:
                    model_response = self.responses[key][model_name][index]

                    model_bleu = self.calculate_bleu(reference, model_response)
                    model_rouge = self.calculate_rouge(reference, model_response)

                    self.models[model_name][key]["bleu"].append(model_bleu)

                    for rouge_metric in ['rouge1', 'rouge2', 'rougeL']:
                        self.models[model_name][key][rouge_metric].append(model_rouge[rouge_metric].fmeasure)

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/scores/model_metric_scores.json", "w") as out:
            json.dump(self.models, out, indent=4)
