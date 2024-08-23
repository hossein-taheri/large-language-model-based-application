import os
import shutil
import traceback

from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class DataMigrationCommand(BaseCommand):
    def __init__(
            self,
            source_dataset_path="creating_dataset/data/qa_raw_data/",
            dataset_path="fine_tuning_using_open_ai_api/data/",
    ):
        super().__init__(name=self.__class__.__name__)
        self.source_dataset_path = source_dataset_path
        self.dataset_path = dataset_path
        self.files = [
            'qa_dataset_train.jsonl',
            'qa_dataset_val.jsonl',
            'qa_dataset_test.jsonl',
            'qa_dataset_unseen_test.jsonl',
        ]

    def setup(self):
        pass

    def execute(self):
        for file in self.files:
            try:
                os.remove(
                    os.path.join(self.dataset_path, file)
                )
            except:
                print(traceback.format_exc())
                pass
            try:
                shutil.copy(
                    os.path.join(self.source_dataset_path, file),
                    os.path.join(self.dataset_path, file)
                )
            except:
                print(traceback.format_exc())
                pass
        return

    def cleanup(self):
        pass
