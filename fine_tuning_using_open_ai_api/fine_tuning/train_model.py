from openai import OpenAI


def create_open_ai_client():
    client = OpenAI()
    return client


def upload_file(client, dataset_path):
    file = client.files.create(
        file=open(dataset_path, "rb"),
        purpose="fine-tune"
    )
    return file


def fine_tune_model(client, training_file_id, validation_file_id):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id.id,
        validation_file=validation_file_id.id,
        model="gpt-4o-mini-2024-07-18",
    )
    print(response)


def fine_tune_model(
        train_dataset_path,
        validation_dataset_path
):
    client = create_open_ai_client()
    training_file_id = str(upload_file(client=client, dataset_path=train_dataset_path).id)
    validation_file_id = str(upload_file(client=client, dataset_path=validation_dataset_path).id)
    # training_file_id = "file-YxugfcbRJDmUCkrzFV4cAzWC"
    # validation_file_id = "file-sdnMQjNST6kMl7WfqOj14prl"
    print("training_file_id", training_file_id)
    print("validation_file_id", validation_file_id)
    client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-4o-mini-2024-07-18"
    )
