# distutils: language=c++

import ctypes
cimport cython
# from ucb_score_cython_cpp cimport CMinMaxStats, cpp_ucb_score


from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP

cdef class MinMaxStats:
    cdef CMinMaxStats *cmin_max_stats

    def __cinit__(self):
        self.cmin_max_stats = new CMinMaxStats()

    def update(self, float value):
        self.cmin_max_stats.update(value)

    def normalize(self, float value):
        return self.cmin_max_stats.normalize(value)


def ucb_score(list child_visit_count, list child_prior, list child_reward, list child_value, MinMaxStats m, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount):
    cdef vector[float] cchild_visit_count = child_visit_count
    cdef vector[float] cchild_prior = child_prior
    cdef vector[float] cchild_reward = child_reward
    cdef vector[float] cchild_value = child_value
    return cpp_ucb_score(cchild_visit_count, cchild_prior, cchild_reward, cchild_value, m.cmin_max_stats.maximum, m.cmin_max_stats.minimum, total_children_visit_counts, pb_c_base, pb_c_init, discount)
