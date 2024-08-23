import json

import numpy as np
from scipy.stats import ttest_rel

from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class StatisticalTestsCommand(BaseCommand):
    def __init__(self):
        self.metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
        self.base_model = {}
        self.base_model_data = {}
        self.fine_tuned_model = {}
        self.fine_tuned_model_data = {}

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/scores/base_model_metric_scores.json') as f:
            self.base_model_data = json.load(f)

        with open('fine_tuning_using_open_ai_api/data/results/scores/fine_tuned_model_metric_scores.json') as f:
            self.fine_tuned_model_data = json.load(f)

        for key in self.base_model_data:
            self.base_model[key] = {}
            self.fine_tuned_model[key] = {}
            for metric in self.metrics:
                self.base_model[key][metric] = np.array(self.base_model_data[key][metric])
                self.fine_tuned_model[key][metric] = np.array(self.fine_tuned_model_data[key][metric])

    def execute(self):
        for key in self.base_model:
            print(f"Extracting metrics for dataset : {key}")
            for metric in self.metrics:
                t_stat, p_value = ttest_rel(self.fine_tuned_model[key][metric], self.base_model[key][metric])
                print(f"Paired t-test for {metric}: t_stat={t_stat}, p_value={p_value}")
            print("\n\n")

    def cleanup(self):
        pass
