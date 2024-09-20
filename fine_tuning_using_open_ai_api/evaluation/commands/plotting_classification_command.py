import json
import matplotlib.pyplot as plt
from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class PlottingClassificationCommand(BaseCommand):
    def __init__(self):
        super().__init__(name=self.__class__.__name__)
        self.data = None

    def setup(self):
        with open('fine_tuning_using_open_ai_api/data/results/scores/results.json') as f:
            self.data = json.load(f)

    def execute(self):
        models = list(self.data.keys())

        test_accuracies = [self.data[model]["test"]["accuracy"] for model in models]
        unseen_test_accuracies = [self.data[model]["unseen_test"]["accuracy"] for model in models]
        plt.style.use('dark_background')

        plt.figure(figsize=(10, 6))
        plt.bar(models, test_accuracies, label='Accuracy', alpha=0.8, color="r", width=0.2)

        plt.title('Accuracy for Test Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        plt.tight_layout()
        for i, v in enumerate(test_accuracies):
            plt.text(i, v + 0.02, f"{v * 100:.2f}%", ha='center', fontsize=12, fontweight='bold')
        plt.savefig(
            'fine_tuning_using_open_ai_api/data/results/charts/test_dataset_comparison.png'
        )
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.bar(models, unseen_test_accuracies, label='Accuracy', alpha=0.8, color='r', width=0.2)

        plt.title('Accuracy for Unseen Test Dataset')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.legend(loc='upper right')
        for i, v in enumerate(unseen_test_accuracies):
            plt.text(i, v + 0.02, f"{v * 100:.2f}%", ha='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            'fine_tuning_using_open_ai_api/data/results/charts/unseen_test_dataset_comparison.png'
        )
        plt.close()

    def cleanup(self):
        plt.close()
