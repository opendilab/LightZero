# distutils:language=c++
# cython:language_level=3
from libcpp.vector cimport vector


cdef extern from "lib/cminimax.cpp":
    pass


cdef extern from "lib/cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        int c_visit
        float c_scale
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void set_static_val(float value_delta_max, int c_visit, float c_scale)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)
        void set_static_val(float value_delta_max, int c_visit, float c_scale)

cdef extern from "lib/cnode.cpp":
    pass


cdef extern from "lib/cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, vector[int] & legal_actions) except +
        int visit_count, to_play, current_latent_state_index, batch_index, best_action
        float value_prefixs, prior, value_sum, parent_value_prefix

        void expand(int to_play, int current_latent_state_index, int batch_index, float value_prefixs,
                    vector[float] policy_logits)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float compute_mean_q(int isRoot, float parent_q, float discount_factor)

        int expanded()
        float value()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        CNode * get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, vector[vector[int]] legal_actions_list) except +
        int root_num
        vector[CNode] roots

        void prepare(float root_noise_weight, const vector[vector[float]] & noises,
                     const vector[float] & value_prefixs, const vector[vector[float]] & policies,
                     vector[int] to_play_batch)
        void prepare_no_noise(const vector[float] & value_prefixs, const vector[vector[float]] & policies,
                              vector[int] to_play_batch)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[float] get_values()
        vector[vector[float]] get_root_policies(CMinMaxStatsList *min_max_stats_lst)
        vector[int] get_best_actions()
        # visualize related code
        # CNode* get_root(int index)

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, search_lens
        vector[int] virtual_to_play_batchs
        vector[CNode *] nodes

    cdef void cbackpropagate(vector[CNode *] & search_path, CMinMaxStats & min_max_stats,
                              int to_play, float value, float discount_factor)
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, vector[float] value_prefixs,
                               vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults & results,
                               vector[int] is_reset_list, vector[int] & to_play_batch)
    void cbatch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, vector[float] value_prefixs, 
                                vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, 
                               vector[int] is_reset_list, vector[int] &to_play_batch, vector[int] &no_inference_lst, 
                               vector[int] &reuse_lst, vector[float] &reuse_value_lst)
    # ========== MuZero/UCB 风格的遍历（备份） ==========
    void cbatch_traverse_ucb(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor,
                             CMinMaxStatsList *min_max_stats_lst, CSearchResults & results,
                             vector[int] & virtual_to_play_batch)
    void cbatch_traverse_with_reuse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor,
                             CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                             vector[int] &virtual_to_play_batch, vector[int] &true_action, vector[float] &reuse_value)

    # ========== EfficientZero V2 风格的遍历（Sequential Halving 集成） ==========
    void cbatch_traverse(CRoots *roots, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                         int num_simulations, int simulation_idx, const vector[vector[float]]& gumble_noise,
                         int current_num_top_actions, vector[int] &virtual_to_play_batch)

    # ========== EfficientZero V2 Sequential Halving ==========
    vector[int] c_batch_sequential_halving(CRoots *roots, vector[vector[float]] gumbel_noises,
                                           CMinMaxStatsList *min_max_stats_lst, int current_phase,
                                           int current_num_top_actions)

cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

cdef class ResultsWrapper:
    cdef CSearchResults cresults

cdef class Roots:
    cdef readonly int root_num
    cdef CRoots *roots
    cdef readonly int num_actions
    cdef public object legal_actions_list

cdef class Node:
    cdef CNode cnode
