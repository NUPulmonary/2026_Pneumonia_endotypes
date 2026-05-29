import argparse
import itertools
import logging
import subprocess


logging.basicConfig(level=logging.INFO)


EPOCH_NB = 20
NB_OF_TRIES = 5

PRETRAINED_PATH = '../data/01_ehrformer/models/pretrained'

PERTURBATION_GROUP_TASK_NAME = 'perturbation_groups_2'

SCRIPT_NAME = 'ehrformer_finetunning_script.py'


PATHOGEN_LIST = [
    'pathogen_all',
    'Early SARS-CoV-2; Gram+',
    'Late SARS-CoV-2',
    'Late SARS-CoV-2; Gram+',
    'Early SARS-CoV-2',
    'Pseudomonas aeruginosa; SARS-CoV-2',
    'Gram-*; Gram+',
    'Gram-*',
    'Gram+',
    'Pseudomonas aeruginosa',
]


PATHOGEN_SPLIT_NAME = [
    'test_split_pg2_split_1',
    'test_split_pg2_split_2',
    'test_split_pg2_split_3',
    'test_split_pg2_split_4',
]

VAP_DICT = {
    'vap_onset_d7_empirical' : ['vap_onset_emp_test_split_1', 'vap_onset_emp_test_split_2', 'vap_onset_emp_test_split_3', 'vap_onset_emp_test_split_4'],
    'is_episode_cured' : ['vap_outcome_test_split_1', 'vap_outcome_test_split_2', 'vap_outcome_test_split_3', 'vap_outcome_test_split_4'],
}


def get_partial_command(
    pretrained_dir: str,
    epoch_nb: int,
):
    def _result(task: str, split: str, category: str, run_nb: int = 0, mc_samples: int = 0):
        command = f'python {SCRIPT_NAME} '
        command += f'--selected_category "{category}" '
        command += f'--split_name "{split}" '
        command += f'--task_name "{task}" '
        command += f'--run_nb "{run_nb}" '
        command += f'--pretrained_path "{pretrained_dir}/{split}" '
        command += f'--epoch_nb "{epoch_nb}" '
        return command
    return _result


if __name__ == '__main__':
    partial_command = get_partial_command(
        pretrained_dir=PRETRAINED_PATH,
        epoch_nb=EPOCH_NB,
    )
    # PATHOGEN PREDICTION
    logging.info(f'##### PATHOGEN PREDICTION #####')
    for category, split in itertools.product(PATHOGEN_LIST, PATHOGEN_SPLIT_NAME):
        for run_nb in range(NB_OF_TRIES):
            current_command = partial_command(
                task=PERTURBATION_GROUP_TASK_NAME,
                split=split,
                category=category,
                run_nb=run_nb + 1
            )
            logging.info(f'Running command: {current_command}')
            subprocess_result = subprocess.call(current_command, shell=True)
            if subprocess_result != 0:
                logging.info(f'Command {current_command} failed.')

    # VAP TASKS PREDICTION
    logging.info(f'##### VAP TASKS PREDICTION #####')
    for category, splits in VAP_DICT.items():
        for split in splits:
            for run_nb in range(NB_OF_TRIES):
                current_command = partial_command(
                    task=category,
                    split=split,
                    category=category,
                    run_nb=run_nb + 1
                )
                logging.info(f'Running command: {current_command}')
                subprocess_result = subprocess.call(current_command, shell=True)
                if subprocess_result != 0:
                    logging.info(f'Command {current_command} failed.')
