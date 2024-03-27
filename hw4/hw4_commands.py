import subprocess
import sys

# Define the commands for each question
commands = {
    'Testing': [
        'python run_hw4_gc.py env.env_name=reacher env.exp_name=q1_reacher alg.rl_alg=pg'
    ],
    'Question 1': [
        'python run_hw4_gc.py env.env_name=reacher env.exp_name=q1_reacher',
        'python run_hw4_gc.py env.env_name=antmaze env.exp_name=q1_ant env.goal_dist=normal'
    ],
    'Question 2': [
        'python run_hw4_gc.py env.env_name=reacher env.exp_name=q1_reacher_normal env.goal_dist=normal',
        'python run_hw4_gc.py env.env_name=antmaze env.exp_name=q1_ant_normal env.goal_dist=normal'
    ],
    'Question 3': [
        'python run_hw4_gc.py env.env_name=reacher env.exp_name=q3_reacher_normal_relative env.goal_dist=normal env.goal_rep=relative',
        'python run_hw4_gc.py env.env_name=antmaze env.exp_name=q3_ant_normal_relative env.goal_dist=normal env.goal_rep=relative'
    ],
    'Question 4': [
        'python run_hw4_gc.py env.env_name=reacher env.exp_name=q3_reacher_normal env.goal_dist=normal env.goal_frequency=10',
        'python run_hw4_gc.py env.env_name=antmaze env.exp_name=q3_ant_normal env.goal_dist=normal env.goal_frequency=10'
    ],
    'Question 5': [
        'python run_hw4.py  env.exp_name=q4_hrl_gf_5 env.l_alg=pg env.env_name=antmaze env.goal_frequency=5',
        'python run_hw4.py  env.exp_name=q4_hrl_gf_7_5 env.l_alg=pg env.env_name=antmaze env.goal_frequency=7.5',
        'python run_hw4.py  env.exp_name=q4_hrl_gf_10 env.rl_alg=pg env.env_name=antmaze env.goal_frequency=10'
    ]
}

def run_commands(question_keys):
    for key in question_keys:
        if key in commands:
            for command in commands[key]:
                print(f"Executing: {command}")
                subprocess.run(command, shell=True)
        else:
            print(f"No commands found for {key}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Get the list of questions to run from command-line arguments
        question_keys = [f'Question {num}' for num in sys.argv[1:]]
        run_commands(question_keys)
    else:
        print("Usage: python file.py [question numbers separated by space]")