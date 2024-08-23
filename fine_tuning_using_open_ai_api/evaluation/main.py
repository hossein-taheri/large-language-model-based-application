import os
import cmd
from fine_tuning_using_open_ai_api.utils import import_env_variables
from fine_tuning_using_open_ai_api.evaluation.commands.extracting_metrics_command import ExtractingMetricsCommand
from fine_tuning_using_open_ai_api.evaluation.commands.plotting_command import PlottingCommand
from fine_tuning_using_open_ai_api.evaluation.commands.response_gathering_command import ResponseGathering
from fine_tuning_using_open_ai_api.evaluation.commands.statistical_tests_command import StatisticalTestsCommand


class Evaluation(cmd.Cmd):
    intro = 'Evaluation. Type "help" for available commands.'

    def do_response_gathering(self, line):
        ResponseGathering().run()

    def do_extracting_metrics(self, line):
        ExtractingMetricsCommand().run()

    def do_statistical_tests(self, line):
        StatisticalTestsCommand().run()

    def do_plotting(self, line):
        PlottingCommand().run()

    def do_clear(self, line):
        os.system('cls' if os.name == 'nt' else 'clear')

    def do_exit(self, line):
        return True


if __name__ == '__main__':
    Evaluation().cmdloop()
