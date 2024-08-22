import os
import cmd
from fine_tuning_using_open_ai_api.utils import import_env_variables
from fine_tuning_using_open_ai_api.fine_tuning.commands.data_migration_command import DataMigrationCommand
from fine_tuning_using_open_ai_api.fine_tuning.commands.price_estimation_command import PriceEstimationCommand
from fine_tuning_using_open_ai_api.fine_tuning.commands.train_model_command import UploadingFilesCommand, \
    FineTuningCommand


class FineTuning(cmd.Cmd):
    intro = 'Fine Tuning. Type "help" for available commands.'

    def do_data_migration(self, line):
        DataMigrationCommand().run()

    def do_price_estimation(self, line):
        PriceEstimationCommand().run()

    def do_uploading_files(self, line):
        UploadingFilesCommand().run()

    def do_fine_tuning(self, line):
        FineTuningCommand().run()

    def do_clear(self, line):
        os.system('cls' if os.name == 'nt' else 'clear')

    def do_exit(self, line):
        return True


if __name__ == '__main__':
    FineTuning().cmdloop()
