import json
import tiktoken


def estimate_fine_tuning_cost(jsonl_file):
    encodings = {
        'gpt-3.5-turbo': tiktoken.encoding_for_model("gpt-3.5-turbo"),
        'gpt-4o-mini-2024-07-18': tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18"),
    }
    with open(jsonl_file, 'r') as f:
        examples = [json.loads(line) for line in f]

    total_tokens_per_model = {
        model_name: sum(
            sum(
                len(
                    encodings[model_name].encode(example['messages'][index]['content'])
                ) for index in range(len(example['messages']))
            ) for example in examples
        ) for model_name in encodings
    }

    cost_per_1m_tokens = {
        'gpt-3.5-turbo': 8,
        'gpt-4o-mini-2024-07-18': 3,
    }

    fine_tuning_costs = {
        model: (total_tokens / 1_000_000) * cost_per_1m_tokens[model] for model, total_tokens in
        total_tokens_per_model.items()
    }

    return total_tokens_per_model, fine_tuning_costs


def predict_qa_dataset_price(jsonl_file_path):
    total_tokens, costs = estimate_fine_tuning_cost(jsonl_file_path)
    for model, total_tokens in total_tokens.items():
        print(f"The estimated total tokens of your data in {model} is {total_tokens}")
    print("\n")
    for model, cost in costs.items():
        print(f"The estimated cost of fine-tuning {model} is ${cost:.2f}")
    print("\n")
