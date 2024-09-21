import json
import matplotlib.pyplot as plt
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class PlottingCommand(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.models_data = self.fine_tuned_model_data = None

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/scores/model_metric_scores.json') as f:
            self.models_data = json.load(f)

    def execute(self):
        metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL']
        keys = ["test", "unseen_test"]
        colors = ["blue", "red", "green", "yellow", "white", "orange", "brown"]

        plt.style.use('dark_background')

        for key in keys:
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                for index, model_name in enumerate(self.models_data):
                    if model_name not in ["base_model", "fine_tuned_model"]:
                        continue
                    plt.plot(
                        self.models_data[model_name][key][metric],
                        marker='o',
                        linestyle='-',
                        label=f'{model_name.upper()} Model',
                        color=colors[index],
                    )
                plt.title(f'Comparison of {metric.upper()}', color='white')
                plt.ylim(0, 1)
                plt.xlabel('Index', color='white')
                plt.ylabel(metric.upper(), color='white')
                plt.legend(facecolor='black', edgecolor='white')
                plt.grid(True)
                plt.tight_layout()
                plt.gcf().set_facecolor('black')
                plt.gca().set_facecolor('black')

                plt.savefig(
                    f'fine_tuning_using_open_ai_api/data/results/charts/{key}_dataset_{metric}_metric_comparison_plot.png'
                )

    def cleanup(self):
        plt.close()
