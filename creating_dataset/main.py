import os
import cmd

from creating_dataset.commands.change_datasets_format import ChangeDatasetsFormat
from creating_dataset.commands.combine_all_jsonl_files import CombineAllJsonlFiles
from creating_dataset.commands.generate_qa_datasets import GenerateQADatasets
from creating_dataset.commands.split_datasets import SplitDatasets


class CreatingDataset(cmd.Cmd):
    intro = 'Creating dataset. Type "help" for available commands.'

    def do_generate_qa_datasets_from_raw_data(self, line):
        GenerateQADatasets().run()

    def do_combine_all_jsonl_files(self, line):
        CombineAllJsonlFiles().run()

    def do_change_datasets_format(self, line):
        ChangeDatasetsFormat().run()

    def do_split_dataset(self, line):
        SplitDatasets().run()

    def do_clear(self, line):
        os.system('cls' if os.name == 'nt' else 'clear')

    def do_exit(self, line):
        return True


if __name__ == '__main__':
    CreatingDataset().cmdloop()
