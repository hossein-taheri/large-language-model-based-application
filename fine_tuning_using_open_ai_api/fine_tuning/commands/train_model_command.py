from openai import OpenAI

from fine_tuning_using_open_ai_api.utils.base_command import BaseCommand


class UploadingFilesCommand(BaseCommand):
    def __init__(
            self,
            train_dataset_path='fine_tuning_using_open_ai_api/data/qa_dataset_train.jsonl',
            validation_dataset_path='fine_tuning_using_open_ai_api/data/qa_dataset_val.jsonl'
    ):
        super().__init__(name=self.__class__.__name__)
        self.client = OpenAI()
        self.train_dataset_path = train_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.training_file_id = self.validation_file_id = None

    def setup(self):
        pass

    def execute(self):
        self.training_file_id = str(
            self.upload_file(client=self.client, dataset_path=self.train_dataset_path).id
        )
        self.validation_file_id = str(
            self.upload_file(client=self.client, dataset_path=self.validation_dataset_path).id
        )

    def cleanup(self):
        print("training_file_id", self.training_file_id)
        print("validation_file_id", self.validation_file_id)

    @staticmethod
    def upload_file(client, dataset_path):
        file = client.files.create(
            file=open(dataset_path, "rb"),
            purpose="fine-tune"
        )
        return file


class FineTuningCommand(BaseCommand):
    def __init__(
            self,
            training_file_id="file-YxugfcbRJDmUCkrzFV4cAzWC",
            validation_file_id="file-sdnMQjNST6kMl7WfqOj14prl"
    ):
        super().__init__(name=self.__class__.__name__)
        self.client = OpenAI()
        self.training_file_id = training_file_id
        self.validation_file_id = validation_file_id

    def setup(self):
        pass

    def execute(self):
        self.client.fine_tuning.jobs.create(
            training_file=self.training_file_id,
            validation_file=self.validation_file_id,
            model="gpt-4o-mini-2024-07-18"
        )

    def cleanup(self):
        pass

    @staticmethod
    def fine_tune_model(client, training_file_id, validation_file_id):
        response = client.fine_tuning.jobs.create(
            training_file=training_file_id.id,
            validation_file=validation_file_id.id,
            model="gpt-4o-mini-2024-07-18",
        )
        print(response)
