import jax.numpy as jnp
from jax import grad, jit, vmap

FLOAT_MAX = 1000000.0
FLOAT_MIN = -FLOAT_MAX


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = FLOAT_MAX
        self.minimum = FLOAT_MIN

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

@jit
def ucb_score_jax_vmap_jit(child_visit_count: float, child_prior: float, child_reward: float, child_value: float,
                       maximum: float, minimum: float, total_children_visit_counts: int, pb_c_base: int, pb_c_init: float,
                       discount: float) -> float:
    pb_c = jnp.log((total_children_visit_counts + pb_c_base + 1) /
                   pb_c_base) + pb_c_init
    pb_c *= jnp.sqrt(total_children_visit_counts) / (child_visit_count + 1)

    prior_score = pb_c * child_prior
    value_score = child_reward +discount * child_value
    value_score = (value_score - minimum) / (maximum - minimum)
    return prior_score + value_score

def ucb_score_jax_vmap(child_visit_count: float, child_prior: float, child_reward: float, child_value: float,
                       min_max_stats: MinMaxStats, total_children_visit_counts: int, pb_c_base: int, pb_c_init: float,
                       discount: float) -> float:
    pb_c = jnp.log((total_children_visit_counts + pb_c_base + 1) /
                   pb_c_base) + pb_c_init
    pb_c *= jnp.sqrt(total_children_visit_counts) / (child_visit_count + 1)

    prior_score = pb_c * child_prior
    value_score = min_max_stats.normalize(child_reward +discount * child_value)
    return prior_score + value_score

def total_ucb_score_jax_vmap(child_visit_count: list, child_prior: list, child_reward: list, child_value: list,
                       min_max_stats: MinMaxStats, total_children_visit_counts: int, pb_c_base: int, pb_c_init: float,
                       discount: float) -> float:
    for i in range(total_children_visit_counts):
        pb_c = jnp.log((total_children_visit_counts + pb_c_base + 1) /
                    pb_c_base) + pb_c_init
        pb_c *= jnp.sqrt(total_children_visit_counts) / (child_visit_count[i] + 1)

        prior_score = pb_c * child_prior[i]
        value_score = min_max_stats.normalize(child_reward[i] + discount * child_value[i])
    
    return prior_score + value_score
