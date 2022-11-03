# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CNode, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse, CAction
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


cdef class Action:
    cdef int is_root_action
    cdef vector[float] value
    cdef CAction action

    def __cinit__(self):
        pass

    def __cinit__(self, vector[float] value, int is_root_action):
        self.is_root_action = is_root_action
        self.value = value



cdef class Roots:
    cdef int root_num
    cdef int action_space_size
    cdef int num_of_sampled_actions
    cdef CRoots *roots

    def __cinit__(self, int root_num, list legal_actions_list, int action_space_size, int num_of_sampled_actions):
        self.root_num = root_num
        self.action_space_size = action_space_size
        self.num_of_sampled_actions = num_of_sampled_actions
        self.roots = new CRoots(root_num, legal_actions_list, action_space_size, num_of_sampled_actions)

    def prepare(self, float root_exploration_fraction, list noises, list value_prefix_pool, list policy_logits_pool, vector[int] &to_play_batch):
        self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, vector[int] &to_play_batch):
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play_batch)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_sampled_actions(self):
        return self.roots[0].get_sampled_actions()

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

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, vector[int] &legal_actions, int action_space_size, int num_of_sampled_actions):
        pass

    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy)

def batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst, list to_play_batch):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst, to_play_batch)


def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list virtual_to_play_batch):

    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults, virtual_to_play_batch)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.virtual_to_play_batchs
