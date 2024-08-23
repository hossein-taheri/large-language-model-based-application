import json

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class ExtractingMetricsCommand(BaseCommand):
    def __init__(self):
        self.base_model_scores = {}
        self.fine_tuned_model_scores = {}
        self.responses = {}

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/responses.json') as f:
            self.responses = json.load(f)
        for key in self.responses:
            self.base_model_scores[key] = {"bleu": [], "rouge1": [], "rouge2": [], "rougeL": []}
            self.fine_tuned_model_scores[key] = {"bleu": [], "rouge1": [], "rouge2": [], "rougeL": []}

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
                base_model_response = self.responses[key]["base_model"][index]
                fine_tuned_model_response = self.responses[key]["fine_tuned_model"][index]

                base_bleu = self.calculate_bleu(reference, base_model_response)
                fine_tuned_bleu = self.calculate_bleu(reference, fine_tuned_model_response)

                base_rouge = self.calculate_rouge(reference, base_model_response)
                fine_tuned_rouge = self.calculate_rouge(reference, fine_tuned_model_response)

                self.base_model_scores[key]["bleu"].append(base_bleu)
                self.fine_tuned_model_scores[key]["bleu"].append(fine_tuned_bleu)
                print(base_bleu, fine_tuned_bleu)
                for rouge_metric in ['rouge1', 'rouge2', 'rougeL']:
                    self.base_model_scores[key][rouge_metric].append(base_rouge[rouge_metric].fmeasure)
                    self.fine_tuned_model_scores[key][rouge_metric].append(fine_tuned_rouge[rouge_metric].fmeasure)

    def cleanup(self):
        with open("fine_tuning_using_open_ai_api/data/results/scores/base_model_metric_scores.json", "w") as out:
            json.dump(self.base_model_scores, out, indent=4)

        with open("fine_tuning_using_open_ai_api/data/results/scores/fine_tuned_model_metric_scores.json", "w") as out:
            json.dump(self.fine_tuned_model_scores, out, indent=4)
