from datetime import datetime


def get_image(path):
    from os import path as os_path

    import pygame
    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def generate_gif_filename(prefix="go", extension=".gif"):
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}{extension}"
    return filename


def flatten_action_to_gtp_action(flatten_action, board_size):
    if flatten_action == board_size * board_size:
        return "pass"

    row = board_size - 1 - (flatten_action // board_size)
    col = flatten_action % board_size

    # 跳过字母 'I'
    if col >= ord('I') - ord('A'):
        col += 1

    col_str = chr(col + ord('A'))
    row_str = str(row + 1)

    gtp_action = col_str + row_str
    return gtp_action
