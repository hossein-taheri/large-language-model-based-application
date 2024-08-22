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

        for metric in self.metrics:
            self.base_model[metric] = np.array(self.data1[metric])
            self.fine_tuned_model[metric] = np.array(self.data2[metric])

    def execute(self):
        for metric in self.metrics:
            t_stat, p_value = ttest_rel(self.base_model[metric], self.fine_tuned_model[metric])

            print(f"{metric}: T-Test : {t_stat}, P-value: {p_value}")

    def cleanup(self):
        pass
