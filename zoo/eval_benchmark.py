import os
import re
import subprocess

# Define the root path of the zoo directory.
ZOO_PATH = './'

# Define the threshold list for the eval_episode_return_mean values.
THRESHOLD_LIST = {
    'cartpole_muzero': 200.0,  # Example threshold for cartpole_muzero
    'cartpole_unizero': 200.0,  # Example threshold for cartpole_unizero
    'atari_muzero': 18.0,  # Example threshold for atari_muzero (env is Pong by default)
    'atari_unizero': 18.0,  # Example threshold for atari_unizero (env is Pong by default)
    'dmc2gym_state_sampled_muzero': 700.0,  # Example threshold for atari_unizero (env is cartpole-swingup by default)
    'dmc2gym_state_sampled_unizero': 700.0,  # Example threshold for atari_unizero (env is cartpole-swingup by default)

    # Add more algorithms and their thresholds as needed
}

# Define the environment and algorithm list for testing.
ENV_ALGO_LIST = [
    {'env': 'cartpole', 'algo': 'muzero'},
    {'env': 'cartpole', 'algo': 'unizero'},
    {'env': 'atari', 'algo': 'muzero'},
    {'env': 'atari', 'algo': 'unizero'},
    {'env': 'dmc2gym_state', 'algo': 'sampled_muzero'},
    {'env': 'dmc2gym_state', 'algo': 'sampled_unizero'},
    # Add more environment and algorithm pairs as needed
]

# Define the evaluator log file name to look for.
EVALUATOR_LOG_FILE = 'evaluator_logger.txt'

# Define the summary log file to store results.
SUMMARY_LOG_FILE = 'benchmark_summary.txt'


def find_config(env: str, algo: str) -> str:
    """
    Recursively search for the config file in the zoo directory for the given environment and algorithm.

    Args:
        env (str): The environment name (e.g., 'cartpole').
        algo (str): The algorithm name (e.g., 'muzero').

    Returns:
        str: The path to the config file if found, otherwise None.
    """
    for root, dirs, files in os.walk(ZOO_PATH):
        # Check if the current directory matches the environment and contains a 'config' directory.
        if env in root and 'config' in dirs:
            config_dir = os.path.join(root, 'config')
            for file in os.listdir(config_dir):
                if env + '_' + algo + '_config' in file and file.endswith('.py'):
                    print(f'[INFO] Found config file: {file}')
                    return os.path.join(config_dir, file)
    return None

def run_algorithm_with_config(config_file: str) -> None:
    """
    Run the algorithm using the specified config file.

    Args:
        config_file (str): The path to the config file.

    Returns:
        None
    """
    # Obtain the directory and file name of the config file
    config_dir = os.path.dirname(config_file)
    config_filename = os.path.basename(config_file)

    # Save the current working directory
    original_dir = os.getcwd()

    try:
        # Change to the directory of the config file
        os.chdir(config_dir)
        # Build the command to run the algorithm
        command = f"python {config_filename}"
        # Run the command and capture any errors
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error occurred while running the algorithm: {e}")
    finally:
        # Change back to the original working directory
        os.chdir(original_dir)

def find_evaluator_log_path(algo: str, env: str) -> str:
    """
    Recursively search for the path of the 'evaluator_logger.txt' file generated during the algorithm's run,
    and select the most recent directory.
    If the directory is in the format '_seed<x>_<y>', extract <y> and choose the largest value; if it's in the format '_seed<x>',
    extract <x>.

    Args:
        algo (str): The algorithm name (e.g., 'cartpole_muzero').
        env (str): The environment name (e.g., 'cartpole').

    Returns:
        str: The path to the 'evaluator_logger.txt' file, or None if not found.
    """
    latest_number = -1
    selected_log_path = None

    # Regular expression to match '_seed<x>' or '_seed<x>_<y>' format
    seed_pattern = re.compile(r'_seed(\d+)(?:_(\d+))?')

    for root, dirs, files in os.walk(ZOO_PATH):
        # Check if the directory path contains the algorithm name and environment name
        if f'data_{algo}' in root and env in root:
            # Look for the 'evaluator_logger.txt' file in the directory
            if EVALUATOR_LOG_FILE in files:
                # Find the '_seed<x>' or '_seed<x>_<y>' part in the directory and extract numbers
                seed_match = seed_pattern.search(root)
                if seed_match:
                    x_value = int(seed_match.group(1))  # Extract <x>
                    y_value = seed_match.group(2)  # Extract <y>, may be None
                    if y_value:
                        number = int(y_value)  # If <y> exists, use <y> for comparison
                    else:
                        number = x_value  # If no <y>, use <x> for comparison

                    # Update to the latest number and record the corresponding log file path
                    if number > latest_number:
                        latest_number = number
                        selected_log_path = os.path.join(root, EVALUATOR_LOG_FILE)

    if selected_log_path:
        print(f'[INFO] Found latest evaluator log file: {selected_log_path}')
        return selected_log_path
    else:
        print('[INFO] No evaluator log file found.')
        return None

def parse_eval_return_mean(log_file_path: str) -> float:
    """
    Parse the eval_episode_return_mean from the evaluator log file.

    Args:
        log_file_path (str): The path to the evaluator log file.

    Returns:
        float: The eval_episode_return_mean as a float, or None if not found.
    """
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

        for i, line in enumerate(lines):
            if 'eval_episode_return_mean' in line:
                if i + 2 < len(lines):
                    next_line = lines[i + 2]
                    parts = next_line.split('|')
                    if len(parts) >= 4:
                        try:
                            return float(parts[3].strip())
                        except ValueError:
                            print(f"[ERROR] Failed to convert {parts[3].strip()} to float.")
                            return None
    return None

def eval_benchmark() -> None:
    """
    Run the benchmark test, log each result, and output a summary table.

    This function will:
    - Search for the configuration file for each environment and algorithm pair,
    - Execute the algorithm,
    - Locate and parse the evaluator log file to extract the `eval_episode_return_mean`,
    - Compare it with the predefined threshold and determine if the test passed or failed,
    - Log the results to a summary file.

    Returns:
        None
    """
    summary_data = []
    passed_count = 0
    failed_count = 0

    for item in ENV_ALGO_LIST:
        env = item['env']
        algo = item['algo']
        print(f"[INFO] Testing {algo} in {env}...")

        # Step 1: Find the config file
        # NOTE: for environments with specific configurations, add custom cases here
        if env == 'dmc2gym_state' and algo == 'sampled_muzero':
            config_file = './dmc2gym/config/dmc2gym_state_sampled_muzero_config.py'
        elif env == 'dmc2gym_state' and algo == 'sampled_unizero':
            config_file = './dmc2gym/config/dmc2gym_state_sampled_unizero_config.py'
        else:
            config_file = find_config(env, algo)
        if config_file is None:
            print(f"[WARNING] Config file for {algo} in {env} not found. Skipping...")
            summary_data.append((env, algo, 'N/A', 'N/A', 'Skipped'))
            continue

        # Step 2: Run the algorithm with the found config file
        # run_algorithm_with_config(config_file)

        # Step 3: Find the evaluator log file
        # NOTE: for environments with specific configurations, add custom cases here
        if env == 'dmc2gym_state' and algo == 'sampled_muzero':
            log_file_path = find_evaluator_log_path('sampled_muzero', 'cartpole-swingup')
        elif env == 'dmc2gym_state' and algo == 'sampled_unizero':
            log_file_path = find_evaluator_log_path('sampled_unizero', 'cartpole-swingup')
        else:
            log_file_path = find_evaluator_log_path(algo, env)

        if log_file_path is None:
            print(f"[WARNING] Evaluator log file for {algo} in {env} not found. Skipping...")
            summary_data.append((env, algo, 'N/A', 'N/A', 'Skipped'))
            continue

        # Step 4: Parse the evaluator log file to get eval_episode_return_mean
        eval_return_mean = parse_eval_return_mean(log_file_path)
        if eval_return_mean is None:
            print(f"[ERROR] Failed to parse eval_episode_return_mean for {algo} in {env}.")
            summary_data.append((env, algo, 'N/A', 'N/A', 'Failed to parse'))
            continue

        # Step 5: Compare the eval_episode_return_mean with the threshold
        threshold = THRESHOLD_LIST.get(env+'_'+algo, float('inf'))
        if eval_return_mean > threshold:
            result = 'Passed'
            passed_count += 1
        else:
            result = 'Failed'
            failed_count += 1

        print(f"[INFO] {result} for {algo} in {env}. Eval mean return: {eval_return_mean}, Threshold: {threshold}")
        summary_data.append((env, algo, eval_return_mean, threshold, result))

    # Print summary table
    print("\n[RESULTS] Benchmark Summary Table")
    print(f"{'Environment':<20}{'Algorithm':<20}{'Eval Return Mean':<20}{'Threshold':<20}{'Result'}")
    for row in summary_data:
        print(f"{row[0]:<20}{row[1]:<20}{row[2]:<20}{row[3]:<20}{row[4]}")

    print(f"\n[SUMMARY] Total Passed: {passed_count}, Total Failed: {failed_count}")

    # Save results to a log file
    with open(SUMMARY_LOG_FILE, 'w') as summary_file:
        summary_file.write("[RESULTS] Benchmark Summary Table\n")
        summary_file.write(f"{'Environment':<20}{'Algorithm':<20}{'Eval Return Mean':<20}{'Threshold':<20}{'Result'}\n")
        for row in summary_data:
            summary_file.write(f"{row[0]:<20}{row[1]:<20}{row[2]:<20}{row[3]:<20}{row[4]}\n")
        summary_file.write(f"\n[SUMMARY] Total Passed: {passed_count}, Total Failed: {failed_count}\n")


if __name__ == "__main__":
    """
    This script automates the process of benchmarking LightZero algorithms across different environments by:
        - Searching for algorithm configuration files,
        - Running the algorithms,
        - Parsing log files for key performance metrics, and
        - Comparing results to predefined thresholds.
    It only supports [sequential] execution and generates a detailed log of the benchmarking results, 
    making it a useful tool for testing and evaluating different algorithms in a standardized manner.
    """
    eval_benchmark()