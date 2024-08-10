import json


def change_prompt_answer_format_to_user_assistant_format(prompt_answer_dataset):
    with open(prompt_answer_dataset, 'r') as file:
        lines = file.readlines()

    user_assistant_dataset = []
    for prompt_answer_pair in lines:
        prompt_answer_pair = json.loads(prompt_answer_pair)
        user_assistant_dataset.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are an empathetic medical assistant to help detection of disease.",
                },
                {
                    "role": "user",
                    "content": prompt_answer_pair['prompt'],
                },
                {
                    "role": "assistant",
                    "content": prompt_answer_pair['completion'],
                },
            ]
        })

    with open(prompt_answer_dataset, 'w') as file:
        for item in user_assistant_dataset:
            json.dump(item, file)
            file.write('\n')
