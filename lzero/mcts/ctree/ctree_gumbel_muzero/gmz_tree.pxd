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
        CNode(float prior, vector[int] &legal_actions) except +
        int visit_count, to_play, hidden_state_index_x, hidden_state_index_y, best_action
        float value_prefixs, prior, value_sum, parent_value_prefix
        vector[CNode]* ptr_node_pool;

        void setdata(float reward, int visit_count, float value_sum, float raw_value, int action_num, vector[float] child_prior, vector[float] child_reward, vector[int] child_visit_count, vector[float] child_value_sum, vector[float]child_raw_value)
        void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefixs, vector[float] policy_logits)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float compute_mean_q(int isRoot, float parent_q, float discount)

        int expanded()
        float value()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        vector[float] get_policy(float discount, int action_space_size)
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, vector[vector[int]] legal_actions_list) except +
        int root_num
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_noise_weight, const vector[vector[float]] &noises, const vector[float] &value_prefixs, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void prepare_no_noise(const vector[float] &value_prefixs, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[vector[float]] get_policies(float discount, int action_space_size)
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens
        vector[int] virtual_to_play_batchs
        vector[CNode*] nodes

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount)
    void cbatch_back_propagate(int hidden_state_index_x, float discount, vector[float] value_prefixs, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] &to_play_batch)
    void cbatch_traverse(CRoots *roots, int num_simulations, int max_num_considered_actions, float discount, CSearchResults &results, vector[int] &virtual_to_play_batch)
    
    cdef int cselect_root_child(CNode* root, float discount, int num_simulations, int max_num_considered_actions)
    cdef int cselect_interior_child(CNode* root, float discount)

    cdef void csoftmax(vector[float] &input, int input_len)
    cdef float compute_mixed_value(float raw_value, vector[float] &q_values, vector[int] &child_visit, vector[float] &child_prior)
    cdef void rescale_qvalues(vector[float] &value, float epsilon)
    cdef vector[float] qtransform_completed_by_mix_value(CNode *root, vector[int] & child_visit, vector[float] & child_prior, float discount, float maxvisit_init, float value_scale, bool rescale_values, float epsilon)
    cdef vector[int] get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations)
    cdef vector[vector[int]] get_table_of_considered_visits(int max_num_considered_actions, int num_simulations)
    cdef vector[float] score_considered(int considered_visit, vector[float] gumbel, vector[float] logits, vector[float] normalized_qvalues, vector[int] visit_counts)
    cdef vector[float] generate_gumbel(float gumbel_scale, float gumbel_rng, int shape)