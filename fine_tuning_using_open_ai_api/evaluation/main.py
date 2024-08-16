import cmd

from fine_tuning_using_open_ai_api.evaluation.commands.extracting_metrics_command import ExtractingMetricsCommand
from fine_tuning_using_open_ai_api.evaluation.commands.plotting_command import PlottingCommand
from fine_tuning_using_open_ai_api.evaluation.commands.response_gathering_command import ResponseGathering


class Evaluation(cmd.Cmd):
    intro = 'Evaluation. Type "help" for available commands.'

    def do_response_gathering(self, line):
        ResponseGathering().run()

    def do_extracting_metrics(self, line):
        ExtractingMetricsCommand().run()

    def do_plotting(self, line):
        PlottingCommand().run()

    def do_quit(self, line):
        return True


if __name__ == '__main__':
    Evaluation().cmdloop()
