"""
This is an illustrative example of Python interfacing with a MCTS (Monte Carlo Tree Search) object implemented in C++.
Please note that this code is not designed for actual execution.
It serves as a conceptual demonstration, providing an understanding of how Python can interact with C++ objects,
specifically within the context of MCTS.
"""
import sys

import torch

sys.path.append('./LightZero/lzero/mcts/ctree/ctree_gumbel_alphazero/build')

import mcts_gumbel_alphazero
mcts_gumbel_alphazero = mcts_gumbel_alphazero.MCTS()

def _policy_value_fn(self, env: 'Env') -> Tuple[Dict[int, np.ndarray], float]:  # noqa
    legal_actions = env.legal_actions
    current_state, current_state_scale = env.current_state()
    current_state_scale = torch.from_numpy(current_state_scale).to(
        device=self._device, dtype=torch.float
    ).unsqueeze(0)
    with torch.no_grad():
        action_probs, value = self._policy_model.compute_policy_value(current_state_scale)
    action_probs_dict = dict(zip(legal_actions, action_probs.squeeze(0)[legal_actions].detach().cpu().numpy()))
    return action_probs_dict, value.item()

action, mcts_probs = mcts_gumbel_alphazero.get_next_action(
    simulate_env=simulate_env,
    policy_value_func=_policy_value_fn,
    temperature=1,
    sample=True,
)

print(action, mcts_probs)