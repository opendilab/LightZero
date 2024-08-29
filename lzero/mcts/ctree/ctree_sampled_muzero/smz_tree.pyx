# distutils:language=c++
# cython:language_level=3
from libcpp.vector cimport vector

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
    cdef bool continuous_action_space

    def __cinit__(self):
        pass

    def __cinit__(self, int root_num, list legal_actions_list, int action_space_size, int num_of_sampled_actions,
                  bool continuous_action_space):
        #def __cinit__(self, int root_num, list legal_actions_list, int action_space_size, int num_of_sampled_actions):
        self.root_num = root_num
        self.action_space_size = action_space_size
        self.num_of_sampled_actions = num_of_sampled_actions
        self.roots = new CRoots(root_num, legal_actions_list, action_space_size, num_of_sampled_actions,
                                continuous_action_space)

    def prepare(self, float root_noise_weight, list noises, list reward_pool, list policy_logits_pool,
                vector[int] & to_play_batch):
        self.roots[0].prepare(root_noise_weight, noises, reward_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list reward_pool, list policy_logits_pool, vector[int] & to_play_batch):
        self.roots[0].prepare_no_noise(reward_pool, policy_logits_pool, to_play_batch)

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
    cdef bool continuous_action_space

    def __cinit__(self):
        pass

    #def __cinit__(self, float prior, vector[int] &legal_actions, int action_space_size, int num_of_sampled_actions):
    def __cinit__(self, float prior, vector[int] & legal_actions, int action_space_size, int num_of_sampled_actions,
                  bool continuous_action_space):
        pass

    def expand(self, int to_play, int current_latent_state_index, int batch_index, float reward,
               list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, current_latent_state_index, batch_index, reward, cpolicy)

def batch_backpropagate(int current_latent_state_index, float discount_factor, list rewards, list values, list policies,
                         MinMaxStatsList min_max_stats_lst, ResultsWrapper results,
                         list to_play_batch):
    cdef int i
    cdef vector[float] crewards = rewards
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_backpropagate(current_latent_state_index, discount_factor, crewards, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, to_play_batch)

def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount_factor, MinMaxStatsList min_max_stats_lst,
                   ResultsWrapper results, list virtual_to_play_batch, bool continuous_action_space):
    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst.cmin_max_stats_lst, results.cresults,
                    virtual_to_play_batch, continuous_action_space)

    return results.cresults.latent_state_index_in_search_path, results.cresults.latent_state_index_in_batch, results.cresults.last_actions, results.cresults.virtual_to_play_batchs
