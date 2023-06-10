# distutils:language=c++
# cython:language_level=3
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../common_lib/cminimax.cpp":
    pass


cdef extern from "../common_lib/cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)

cdef extern from "lib/cnode.cpp":
    pass


cdef extern from "lib/cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, vector[int] &legal_actions, bool is_chance, int chance_space_size) except +
        int visit_count, to_play, current_latent_state_index, batch_index, best_action
        float value_prefixs, prior, value_sum, parent_value_prefix

        void expand(int to_play, int current_latent_state_index, int batch_index, float value_prefixs, vector[float] policy_logits, bool is_chance)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float compute_mean_q(int isRoot, float parent_q, float discount_factor)

        int expanded()
        float value()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, vector[vector[int]] legal_actions_list, int chance_space_size) except +
        int root_num, chance_space_size
        vector[CNode] roots

        void prepare(float root_noise_weight, const vector[vector[float]] &noises, const vector[float] &value_prefixs, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void prepare_no_noise(const vector[float] &value_prefixs, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, search_lens
        vector[int] virtual_to_play_batchs
        vector[bool] leaf_node_is_chance
        vector[CNode*] nodes

    cdef void cbackpropagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor)
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, vector[float] value_prefixs, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] &to_play_batch, vector[bool] &is_chance_list, vector[int] &leaf_idx_list)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] &virtual_to_play_batch)
