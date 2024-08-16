import json
import matplotlib.pyplot as plt
from fine_tuning_using_open_ai_api.evaluation.commands.base_command import BaseCommand


class PlottingCommand(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/scores/base_model_response.json') as f:
            self.data1 = json.load(f)

        with open('fine_tuning_using_open_ai_api/data/results/scores/fine_tuned_model_scores.json') as f:
            self.data2 = json.load(f)

    def execute(self):
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']

        plt.style.use('dark_background')

        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(self.data1[metric], marker='o', linestyle='-', label='Base Model', color='blue')
            plt.plot(self.data2[metric], marker='o', linestyle='-', label='Fine-Tuned Model', color='red')
            plt.title(f'Comparison of {metric.upper()}', color='white')
            plt.ylim(0, 1)
            plt.xlabel('Index', color='white')
            plt.ylabel(metric.upper(), color='white')
            plt.legend(facecolor='black', edgecolor='white')
            plt.grid(True)
            plt.tight_layout()
            plt.gcf().set_facecolor('black')
            plt.gca().set_facecolor('black')

            plt.savefig(f'fine_tuning_using_open_ai_api/data/results/charts/{metric}_comparison_plot.png')

    def cleanup(self):
        plt.close()
