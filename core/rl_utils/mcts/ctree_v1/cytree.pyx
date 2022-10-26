# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CNode, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


cdef class Roots:
    cdef int root_num
    cdef int pool_size
    cdef CRoots *roots
    cdef vector[int] legal_actions
    cdef vector[vector[int]] legal_actions_list

    def __cinit__(self):
        pass

    def __cinit__(self, int root_num, int action_num, int tree_nodes, list legal_actions=None, list legal_actions_list=None):
        self.root_num = root_num
        self.pool_size = action_num * (tree_nodes + 2)
        if legal_actions != None:
            self.legal_actions = legal_actions
        if legal_actions_list != None:
            self.legal_actions_list = legal_actions_list

    def prepare(self, float root_exploration_fraction, list noises, list value_prefix_pool, list policy_logits_pool, list to_play=None):
        cdef vector[int] tmp = to_play
        if to_play is None:
            self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
        else:
            self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, tmp)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, list to_play=None):
        cdef vector[int] tmp = to_play
        if to_play is None:
            self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool)
        else:
            self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, tmp)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values(self):
        return self.roots[0].get_values()

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


cdef class Node:
    cdef CNode cnode
    cdef vector[int] legal_actions

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num, list ptr_node_pool, list legal_actions=None):
        self.legal_actions = legal_actions
        pass

    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy)

def batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst)


def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list virtual_to_play=None):
    cdef vector[int] tmp = virtual_to_play
    if virtual_to_play is None:
        cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)
    else:
        cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults, tmp)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions
