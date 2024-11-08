import os

def parse_eval_return_mean(log_file_path):
    """
    Parse the eval_episode_return_mean from the evaluator log file, reading from the end of the file.
    :param log_file_path: The path to the evaluator log file.
    :return: The eval_episode_return_mean as a float, or None if not found.
    """
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()

            # Reverse the lines to start reading from the end of the file
            for i, line in enumerate(reversed(lines)):
                if 'eval_episode_return_mean' in line:
                    prev_line = lines[-i + 1]  # This gets the 'next' line (2 lines after in normal order)
                    parts = prev_line.split('|')
                    if len(parts) >= 4:
                        try:
                            return float(parts[3].strip())
                        except ValueError:
                            print(f"[ERROR] Failed to convert {parts[3].strip()} to float in {log_file_path}.")
                            return None
    except FileNotFoundError:
        print(f"[ERROR] Log file {log_file_path} not found.")
    except Exception as e:
        print(f"[ERROR] An error occurred while reading {log_file_path}: {str(e)}")

    return None


def find_evaluator_log_files(base_path):
    """
    Find all evaluator_logger.txt files within the specified base directory.
    :param base_path: The base directory to start searching from.
    :return: A list of paths to evaluator_logger.txt files.
    """
    evaluator_log_paths = []

    # Walk through the base directory recursively
    for root, dirs, files in os.walk(base_path):
        # Check if the current folder contains evaluator_logger.txt
        if 'evaluator_logger.txt' in files:
            # Construct the full path to the evaluator_logger.txt file
            log_file_path = os.path.join(root, 'evaluator_logger.txt')
            evaluator_log_paths.append(log_file_path)

    return evaluator_log_paths


def extract_game_name_from_path(log_file_path):
    """
    Extract the game name from the log file path.
    The game name is assumed to be the part of the path just before '_atari'.
    :param log_file_path: The path to the evaluator log file.
    :return: The extracted game name.
    """
    # Split the path and extract the game name from the folder structure
    try:
        parts = log_file_path.split('/')
        for part in parts:
            if '_atari' in part:
                game_name = part.split('_atari')[0]
                return game_name
    except Exception as e:
        print(f"[ERROR] Couldn't extract game name from {log_file_path}: {str(e)}")
    return None


def get_eval_means_for_games(base_path):
    """
    Get the eval_episode_return_mean for all games under the base directory, along with the game names.
    :param base_path: The path to the base directory containing game logs.
    :return: A tuple of lists: (game_names_list, eval_means_list) and the combined dictionary {game_name: eval_mean}.
    """
    game_names = []
    eval_means = []
    game_eval_dict = {}

    # Find all evaluator_logger.txt files
    log_files = find_evaluator_log_files(base_path)

    # Parse each log file for eval_episode_return_mean and extract the game name
    for log_file in log_files:
        eval_mean = parse_eval_return_mean(log_file)
        game_name = extract_game_name_from_path(log_file)

        if game_name is not None:
            game_names.append(game_name)
            eval_means.append(eval_mean)
            game_eval_dict[game_name] = eval_mean

    return game_names, eval_means, game_eval_dict


def save_results_to_file(game_eval_dict, file_path):
    """
    Save the game names and eval means to a text file.
    :param game_eval_dict: Dictionary of game names and corresponding eval means.
    :param file_path: The path to the output text file.
    """
    try:
        with open(file_path, 'w') as file:
            file.write("Game Names and Eval Episode Return Means:\n")
            file.write("=" * 50 + "\n")
            for game_name, eval_mean in game_eval_dict.items():
                file.write(f"Game: {game_name}, Eval Episode Return Mean: {eval_mean}\n")
        print(f"[INFO] Results saved to {file_path}.")
    except Exception as e:
        print(f"[ERROR] Failed to save the file: {str(e)}")


if __name__ == "__main__":
    base_path = "config/data_muzero"
    game_names, eval_means, game_eval_dict = get_eval_means_for_games(base_path)

    # Display the lists and dictionary in a more readable way
    print("Game Names List:")
    print("=" * 50)
    for game_name in game_names:
        print(game_name)

    print("\nEval Episode Return Means List:")
    print("=" * 50)
    for eval_mean in eval_means:
        print(eval_mean)

    print("\nCombined Dictionary (game_name -> eval_episode_return_mean):")
    print("=" * 50)
    for game_name, eval_mean in game_eval_dict.items():
        print(f"{game_name}: {eval_mean}")

    # Option to save the results to a file
    save_option = input("\nWould you like to save the results to a text file? (y/n): ").strip().lower()
    if save_option == 'y':
        file_path = input("Enter the desired output file path (e.g., results.txt): ").strip()
        save_results_to_file(game_eval_dict, file_path)