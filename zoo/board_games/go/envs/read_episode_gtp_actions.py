
dataset_file_dir = '/Users/puyuan/code/LightZero/go_sgf_dataset/1_result_pos.txt'

def read_episode_gtp_actions_from_txt(dataset_file_dir):
    """
    Overview:
        read episode gtp actions from txt file
    Arguments:
        - dataset_file_dir (:obj:`str`): the directory of the dataset file
    Returns:
        - episode_gtp_actions (:obj:`list`): the list of episode gtp actions
    """
    # initialize an empty list
    episode_gtp_actions = []

    # open file and read the content in a list
    with open(dataset_file_dir, 'r') as file:
        for line in file:
            # remove linebreak, which is the last character of the string
            line = line.rstrip('\n')

            # add item to the list
            episode_gtp_actions.append(line)

    print('episode_gtp_actions: ', episode_gtp_actions)
    print('length of episode_gtp_actions:', len(episode_gtp_actions))
    return episode_gtp_actions

episode_gtp_actions = read_episode_gtp_actions_from_txt(dataset_file_dir)
