# distutils: language=c++
# cython:language_level=3
from libcpp.vector cimport vector
from libcpp cimport bool

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
    cdef CRoots *roots

    def __cinit__(self, int root_num, vector[vector[int]] legal_actions_list, int chance_space_size):
        self.root_num = root_num
        self.roots = new CRoots(root_num, legal_actions_list, chance_space_size)

    def prepare(self, float root_noise_weight, list noises, list value_prefix_pool, list policy_logits_pool,
                vector[int] & to_play_batch):
        self.roots[0].prepare(root_noise_weight, noises, value_prefix_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool, vector[int] & to_play_batch):
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool, to_play_batch)

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

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, vector[int] & legal_actions, bool is_chance, int chance_space_size):
        pass

    def expand(self, int to_play, int current_latent_state_index, int batch_index, float value_prefix,
               list policy_logits, bool is_chance):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, current_latent_state_index, batch_index, value_prefix, cpolicy, is_chance)

def batch_backpropagate(int current_latent_state_index, float discount_factor, list value_prefixs, list values, list policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list to_play_batch, list is_chance_list, list leaf_idx_list):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_backpropagate(current_latent_state_index, discount_factor, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, to_play_batch, is_chance_list, leaf_idx_list)

def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst, results.cresults,
                    virtual_to_play_batch)

    return results.cresults.leaf_node_is_chance, results.cresults.latent_state_index_in_search_path, results.cresults.latent_state_index_in_batch, results.cresults.last_actions, results.cresults.virtual_to_play_batchs

