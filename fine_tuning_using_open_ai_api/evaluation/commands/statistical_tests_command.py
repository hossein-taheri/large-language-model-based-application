import json

import numpy as np
from scipy.stats import ttest_rel

from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class StatisticalTestsCommand(BaseCommand):
    def __init__(self):
        self.metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
        self.base_model = {}
        self.fine_tuned_model = {}

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/scores/base_model_response.json') as f:
            self.data1 = json.load(f)

        with open('fine_tuning_using_open_ai_api/data/results/scores/fine_tuned_model_scores.json') as f:
            self.data2 = json.load(f)

        for key in self.data1:
            self.base_model[key] = {}
            self.fine_tuned_model[key] = {}
            for metric in self.metrics:
                self.base_model[key][metric] = np.array(self.data1[key][metric])
                self.fine_tuned_model[key][metric] = np.array(self.data2[key][metric])

    def execute(self):
        for key in self.base_model:
            print(f"Extracting metrics for dataset : {key}")
            for metric in self.metrics:
                t_stat, p_value = ttest_rel(self.base_model[key][metric], self.fine_tuned_model[key][metric])

                print(f"{metric}: T-Test : {t_stat}, P-value: {p_value}")
            print("\n\n")

    def cleanup(self):
        pass
