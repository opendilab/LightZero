# distutils: language=c++

import ctypes
cimport cython
# from eval_cpp cimport CMinMaxStats, cpp_ucb_score
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP



FLOAT_MAX = 1000000.0
FLOAT_MIN = -FLOAT_MAX


cdef class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""
    cdef float maximum
    cdef float minimum

    def __init__(self):
        self.maximum = FLOAT_MAX
        self.minimum = FLOAT_MIN

    def update(self, float value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, float value) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
        
cdef float cython_ucb_score(vector[float] child_visit_count, vector[float] child_prior, vector[float] child_reward, vector[float] child_value, MinMaxStats min_max_stats, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount):
    cdef float ucb_value = 0.0
    cdef float pb_c
    cdef float prior_score
    cdef float value_score
    for i in range(int(total_children_visit_counts)):
        pb_c = 0.0
        prior_score = 0.0
        value_score = 0.0
        pb_c = np.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= (np.sqrt(total_children_visit_counts) / (child_visit_count[i] + 1))

        prior_score = pb_c * child_prior[i]

        value_score = min_max_stats.normalize(child_reward[i] + discount * child_value[i])

        if (value_score < 0):
            value_score = 0.

        ucb_value = prior_score + value_score
    
    return ucb_value;
    
    
def ucb_score(list child_visit_count, list child_prior, list child_reward, list child_value, MinMaxStats min_max_stats, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount):
    cdef vector[float] cchild_visit_count = child_visit_count
    cdef vector[float] cchild_prior = child_prior
    cdef vector[float] cchild_reward = child_reward
    cdef vector[float] cchild_value = child_value
    return cython_ucb_score(cchild_visit_count, cchild_prior, cchild_reward, cchild_value, min_max_stats, total_children_visit_counts, pb_c_base, pb_c_init, discount)