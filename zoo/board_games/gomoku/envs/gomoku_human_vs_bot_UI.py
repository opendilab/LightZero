import os
import subprocess
import tkinter as tk

import imageio
from PIL import ImageGrab


class GomokuUI(tk.Tk):
    def __init__(self, gomoku_env, save_frames=True):
        tk.Tk.__init__(self)
        self.env = gomoku_env
        self.board_size = gomoku_env.board_size
        self.cell_size = 50
        self.canvas_size = self.cell_size * (self.board_size + 1)

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='peach puff')
        self.canvas.pack()
        self.frames = []
        self.canvas.bind("<Button-1>", self.click)
        self.save_frames = save_frames

    def click(self, event):
        # Adjust the x and y coordinates to account for the boundary
        adjusted_x = event.y - self.cell_size
        adjusted_y = event.x - self.cell_size

        # Map the click to the nearest intersection point
        x = (adjusted_x + self.cell_size // 2) // self.cell_size
        y = (adjusted_y + self.cell_size // 2) // self.cell_size

        action = self.coord_to_action(x, y)
        self.update_board(action, from_ui=True)

    def update_board(self, action=None, from_ui=False):
        if from_ui:
            print('player 1: ' + self.env.action_to_string(action))
            timestep = self.env.step(action)
            self.timestep = timestep
            obs = self.timestep.obs
        else:
            obs = {'board': self.env.board}

        # Update the board UI
        for i in range(0, self.board_size):
            for j in range(0, self.board_size):
                if obs['board'][i][j] == 1:  # black
                    color = 'black'
                    self.draw_piece(i, j, color)
                elif obs['board'][i][j] == 2:  # white
                    color = 'white'
                    self.draw_piece(i, j, color)
                # else:
                #     # only for debug
                #     self.draw_piece(i, j, color)
        if self.save_frames:
            self.save_frame()
        self.update_turn_label()
        # time.sleep(0.1)

        # Check if the game has ended
        if self.timestep.done:
            self.quit()

    def draw_piece(self, x, y, color):
        padding = self.cell_size // 2
        self.canvas.create_oval(y * self.cell_size + padding, x * self.cell_size + padding,
                                (y + 1) * self.cell_size + padding, (x + 1) * self.cell_size + padding, fill=color)

    def save_frame_bkp(self):
        # Get the bounds of the window
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Grab the image and save it
        img = ImageGrab.grab(bbox=(x, y, x1, y1))  # bbox参数定义了截图区域
        img.save("frame.png")

        # Append the image to the frames
        self.frames.append(imageio.imread("frame.png"))

    def save_frame(self):
        # Generate Postscript from the canvas
        ps = self.canvas.postscript(colormode='color')

        # Use ImageMagick to convert the Postscript to PNG
        with open('temp.ps', 'w') as f:
            f.write(ps)
        # subprocess.run(['convert', 'temp.ps', 'frame.png'])
        subprocess.run(['convert', '-colorspace', 'sRGB', 'temp.ps', 'frame.png'])
        os.remove('temp.ps')

        # Append the PNG to the frames
        self.frames.append(imageio.imread('frame.png'))

    def save_gif(self, file_name):
        # Save frames as gif file
        imageio.mimsave(file_name, self.frames, 'GIF', duration=0.1)

    def draw_board(self):
        self.canvas.create_text(self.canvas_size // 2, self.cell_size // 2, text="Gomoku (Human vs AI)",
                                font=("Arial", 10))
        # Reduce the loop count to avoid drawing extra lines
        for i in range(1, self.board_size + 1):
            self.canvas.create_line(i * self.cell_size, self.cell_size, i * self.cell_size,
                                    self.canvas_size - self.cell_size)
            self.canvas.create_line(self.cell_size, i * self.cell_size, self.canvas_size - self.cell_size,
                                    i * self.cell_size)
        self.update_turn_label()

    def update_turn_label(self):
        # Change the label text
        turn_text = "Human's Turn (Black)" if self.env.current_player == 1 else "AI's Turn (White)"
        self.canvas.create_text(self.canvas_size // 2, self.cell_size, text=turn_text, font=("Arial", 10))

    def coord_to_action(self, x, y):
        # Adjusted the coordinate system
        return x * self.board_size + y


from easydict import EasyDict
from zoo.board_games.gomoku.envs.gomoku_env import GomokuEnv


# 测试函数
def test_ui():
    cfg = EasyDict(
        board_size=15,
        battle_mode='play_with_bot_mode',
        prob_random_agent=0,
        channel_last=False,
        scale=True,
        agent_vs_human=False,
        bot_action_type='v1',
        prob_random_action_in_bot=0.,
        render_mode='state_realtime_mode',
        screen_scaling=9,
        check_action_to_connect4_in_bot_v0=False,
        save_frames=True,
        # save_frames=False,
    )
    env = GomokuEnv(cfg)
    env.reset()
    game_ui = GomokuUI(env, save_frames=cfg.save_frames)
    game_ui.draw_board()

    while True:
        game_ui.mainloop()
        if game_ui.timestep.done:
            game_ui.save_gif('gomoku_human_vs_bot.gif')

            if game_ui.timestep.reward != 0 and game_ui.timestep.info['next player to play'] == 2:
                print('player 1 (human player) win')

            elif game_ui.timestep.reward != 0 and game_ui.timestep.info['next player to play'] == 1:
                print('player 2 (AI player) win')
            else:
                print('draw')
            break


if __name__ == "__main__":
    test_ui()
