from __future__ import annotations

from pathlib import Path

import pooltool as pt

from pooltool.ai.bot.sumtothree_rl.core import SumToThreeAI, single_player_env

if __name__ == "__main__":
    model_path = Path("./ckpt/ckpt_best.pth.tar")
    ai = SumToThreeAI.load(model_path)
    gui = pt.ShotViewer()

    while True:
        env = single_player_env(random_pos=True)
        action = ai.decide(env.system, env.game)
        ai.apply(env.system, action)
        env.simulate()
        gui.show(env.system)
