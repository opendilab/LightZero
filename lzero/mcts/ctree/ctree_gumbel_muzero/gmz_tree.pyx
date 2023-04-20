# distutils: language=c++
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


cdef class Roots:
    cdef int root_num
    cdef CRoots *roots

    def __cinit__(self, int root_num, vector[vector[int]] legal_actions_list):
        self.root_num = root_num
        self.roots = new CRoots(root_num, legal_actions_list)

    def prepare(self, float root_noise_weight, list noises, list value_prefix_pool, list value_pool, list policy_logits_pool, vector[int] &to_play_batch):
        self.roots[0].prepare(root_noise_weight, noises, value_prefix_pool, value_pool, policy_logits_pool, to_play_batch)

    def prepare_no_noise(self, list value_prefix_pool, list value_pool, list policy_logits_pool, vector[int] &to_play_batch):
        self.roots[0].prepare_no_noise(value_prefix_pool, value_pool, policy_logits_pool, to_play_batch)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()
    
    def get_policies(self, float discount, int action_space_size):
        return self.roots[0].get_policies(discount, action_space_size)

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

    def __cinit__(self, float prior, vector[int] &legal_actions):
        pass

    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, float value, list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, value, cpolicy)

    def set_data(self, float reward, int visit_count, float value_sum, float raw_value, int action_num, list child_prior, list child_reward, list child_visit_count, list child_value_sum, list child_raw_value):
        cdef vector[float] cchild_prior=child_prior
        cdef vector[float] cchild_reward=child_reward
        cdef vector[int] cchild_visit_count = child_visit_count
        cdef vector[float] cchild_value_sum = child_value_sum
        cdef vector[float] cchild_raw_value = child_raw_value
        self.cnode.setdata(reward, visit_count, value_sum, raw_value, action_num, cchild_prior, cchild_reward, cchild_visit_count, cchild_value_sum, cchild_raw_value)
        

def batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list to_play_batch):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, to_play_batch)


def batch_traverse(Roots roots, int num_simulations, int max_num_considered_actions, float discount, ResultsWrapper results, list virtual_to_play_batch):

    cbatch_traverse(roots.roots, num_simulations, max_num_considered_actions, discount, results.cresults, virtual_to_play_batch)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions, results.cresults.virtual_to_play_batchs

def select_root_child(Node roots, float discount, int num_simulations, int max_num_considered_actions):

    return cselect_root_child(&roots.cnode, discount, num_simulations, max_num_considered_actions)

def select_interior_child(Node roots, float discount):

    return cselect_interior_child(&roots.cnode, discount)

def softmax(list num_list):
    cdef vector[float] cnum_list = num_list;
    cdef int clength = len(num_list)
    csoftmax(cnum_list, clength)
    for i in range(len(num_list)):
        num_list[i] = cnum_list[i]
    return num_list

def pcompute_mixed_value(float raw_value, list q_values, list child_visit, list child_prior):
    cdef vector[float] cq_values = q_values
    cdef vector[int] cchild_visit = child_visit
    cdef vector[float] cchild_prior = child_prior
    return compute_mixed_value(raw_value, cq_values, cchild_visit, cchild_prior)

def prescale_qvalues(list value, float epsilon):
    cdef vector[float] cvalue = value
    rescale_qvalues(cvalue, epsilon)
    for i in range(len(value)):
        value[i] = cvalue[i]
    return value

def pqtransform_completed_by_mix_value(Node roots, list child_visit, child_prior, discount, maxvisit_init, value_scale, rescale_values, epsilon):
    cdef vector[int] cchild_visit=child_visit
    cdef vector[float] cchild_prior=child_prior
    cdef vector[float] cmix_value = qtransform_completed_by_mix_value(&roots.cnode, cchild_visit, cchild_prior, discount, maxvisit_init, value_scale, rescale_values, epsilon)
    mix_value = []
    for i in range(len(child_visit)):
        mix_value.append(cmix_value[i])
    return mix_value

def pget_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations):
    return get_sequence_of_considered_visits(max_num_considered_actions, num_simulations)

def pget_table_of_considered_visits(int max_num_considered_actions, int num_simulations):
    cdef vector[vector[int]] table = get_table_of_considered_visits(max_num_considered_actions, num_simulations)
    result = []
    for i in range(max_num_considered_actions+1):
        result.append(table[i])
    return result

def pscore_considered(int considered_visit, list gumbel, list logits, list normalized_qvalues, list visit_counts):
    cdef vector[float] cgumbel=gumbel
    cdef vector[float] clogits=logits
    cdef vector[float] cnormalized_qvalues=normalized_qvalues
    cdef vector[int] cvisit_counts=visit_counts
    return score_considered(considered_visit, cgumbel, clogits, cnormalized_qvalues, cvisit_counts)

def pgenerate_gumbel(float gumbel_scale, float gumbel_rng, int shape):
    return generate_gumbel(gumbel_scale, gumbel_rng, shape)