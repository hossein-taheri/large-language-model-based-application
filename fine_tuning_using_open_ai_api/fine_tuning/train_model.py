from openai import OpenAI


def create_open_ai_client():
    client = OpenAI()
    return client


def upload_file(client, dataset_path):
    client.fine_tuning.prepare_data(dataset_path)
    client.files.create(
        file=open(dataset_path, "rb"),
        purpose="fine-tune"
    )


def fine_tune_model(dataset_path):
    client = create_open_ai_client()
    # upload_file(client=client, dataset_path=dataset_path)
    response = client.files.list()
    print(response)

    # client.fine_tuning.jobs.create(
    #     training_file="file-UdpoGypfugl5IwrCANWIGmYu",
    #     model="gpt-4o-mini-2024-07-18"
    # )

    # client.fine_tuning.files.list(limit=10)
    # client.fine_tuning.jobs.list(limit=10)
