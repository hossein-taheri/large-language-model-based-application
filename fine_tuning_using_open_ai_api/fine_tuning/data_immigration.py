import os
import shutil


def immigrate_data(source_dataset_path, dataset_path):
    try:
        os.remove(dataset_path)
    except:
        pass
    try:
        shutil.copy(source_dataset_path, dataset_path)
    except:
        pass
    return