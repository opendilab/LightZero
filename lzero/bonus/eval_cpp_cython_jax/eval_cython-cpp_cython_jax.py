"""
Overview:
    Efficiency comparison of different vectorization methods based on function `ucb_score`:
    NOTE: The time may vary on different devices and software versions.
    =======================================
    ### execute ucb_score 1000,000 times###
    ---------------------------------------
    | Methods                 | Seconds
    ---------------------------------------
    | jax                     | 28.745
    | jax + vmap              | 15.052
    | jax + vmap + jit        | 16.903
    | pure cython             | 1.904
    | cython + cpp + openmp   | 0.041
    | cython + cpp            | 0.036

"""

import random

import jax
from jax import vmap

import pyximport
pyximport.install()

from eval_cython_cpp import ucb_score_cython_cpp
from eval_cython_cpp_openmp import ucb_score_cython_cpp_openmp
from eval_cython import ucb_score_cython
from eval_jax.eval_jax import MinMaxStats, ucb_score_jax, ucb_score_jax_vmap, ucb_score_jax_vmap_jit
from ding.utils import EasyTimer

timer = EasyTimer(cuda=True)


def generate_data(parent_visit_count):
    child_count = [float(random.randint(1, 10)) for i in range(parent_visit_count)]
    child_prior = [random.random() for i in range(parent_visit_count)]
    child_reward = [random.random() for i in range(parent_visit_count)]
    child_value = [random.random() for i in range(parent_visit_count)]

    return child_count, child_prior, child_reward, child_value


def eval_jax_time(parent_visit_count, data):
    min_max_stats = MinMaxStats()

    child_count = data[0]
    child_prior = data[1]
    child_reward = data[2]
    child_value = data[3]

    for i in range(parent_visit_count):
        min_max_stats.update(child_value[i])

    # without vmap
    with timer:
        ucb_score_jax(
            child_count,
            child_prior,
            child_reward,
            child_value,
            min_max_stats,
            total_children_visit_counts=parent_visit_count,
            pb_c_base=1.25,
            pb_c_init=19652,
            discount=0.997
        )
    print(f"eval_jax_time: {timer.value}")

    # with vmap
    with timer:
        vmap(
            ucb_score_jax_vmap, in_axes=(0, 0, 0, 0, None, None, None, None, None)
        )(
            jax.numpy.array(child_count), jax.numpy.array(child_prior), jax.numpy.array(child_reward),
            jax.numpy.array(child_value), min_max_stats, parent_visit_count, 1.25, 19652, 0.997
        )
    print(f"eval_jax_vmap_time: {timer.value}")

    # with vmap and jit
    with timer:
        vmap(
            ucb_score_jax_vmap_jit, in_axes=(0, 0, 0, 0, None, None, None, None, None, None)
        )(
            jax.numpy.array(child_count), jax.numpy.array(child_prior), jax.numpy.array(child_reward),
            jax.numpy.array(child_value), min_max_stats.maximum, min_max_stats.minimum, parent_visit_count, 1.25, 19652,
            0.997
        )
    print(f"eval_jax_vmap_jit_time: {timer.value}")


def eval_cython_time(parent_visit_count, data):
    min_max_stats = ucb_score_cython.MinMaxStats()

    child_count = data[0]
    child_prior = data[1]
    child_reward = data[2]
    child_value = data[3]

    for i in range(parent_visit_count):
        min_max_stats.update(child_value[i])

    with timer:
        ucb_score_cython.ucb_score(
            child_count,
            child_prior,
            child_reward,
            child_value,
            min_max_stats,
            total_children_visit_counts=parent_visit_count,
            pb_c_base=1.25,
            pb_c_init=19652,
            discount=0.997
        )
    print(f"eval_cython_time: {timer.value}")


def eval_cython_cpp_time(parent_visit_count, data):
    min_max_stats = ucb_score_cython_cpp.MinMaxStats()

    child_count = data[0]
    child_prior = data[1]
    child_reward = data[2]
    child_value = data[3]

    for i in range(parent_visit_count):
        min_max_stats.update(child_value[i])

    with timer:
        ucb_score_cython_cpp.ucb_score(
            child_count,
            child_prior,
            child_reward,
            child_value,
            min_max_stats,
            total_children_visit_counts=parent_visit_count,
            pb_c_base=1.25,
            pb_c_init=19652,
            discount=0.997
        )
    print(f"eval_cython-cpp_time: {timer.value}")


def eval_cython_cpp_openmp_time(parent_visit_count, data):
    min_max_stats = ucb_score_cython_cpp_openmp.MinMaxStats()

    child_count = data[0]
    child_prior = data[1]
    child_reward = data[2]
    child_value = data[3]

    for i in range(parent_visit_count):
        min_max_stats.update(child_value[i])

    with timer:
        ucb_score_cython_cpp_openmp.ucb_score(
            child_count,
            child_prior,
            child_reward,
            child_value,
            min_max_stats,
            total_children_visit_counts=parent_visit_count,
            pb_c_base=1.25,
            pb_c_init=19652,
            discount=0.997
        )
    print(f"eval_cythoon-cpp-openmp_time: {timer.value}")


if __name__ == "__main__":
    parent_visit_count = 1000000
    data = generate_data(parent_visit_count)

    print("###execute ucb_score 1000,000 times###")

    # eval_jax_time(parent_visit_count, data)
    eval_cython_time(parent_visit_count, data)
    # eval_cython_cpp_openmp_time(parent_visit_count, data)
    # eval_cython_cpp_time(parent_visit_count, data)
