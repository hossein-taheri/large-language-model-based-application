import os
import shutil


def immigrate_data():
    try:
        os.remove(
            'fine_tuning_using_open_ai_api/data/qa_dataset.jsonl'
        )
    except:
        pass
    try:
        shutil.copy(
            'creating_dataset/data/qa_raw_data/qa_dataset.jsonl',
            'fine_tuning_using_open_ai_api/data/qa_dataset.jsonl'
        )
    except:
        pass
