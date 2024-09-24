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
    cdef cppclass CAction:
        CAction() except +
        CAction(vector[float] value, int is_root_action)  except +
        int is_root_action
        vector[float] value

        vector[size_t] get_hash();
        size_t get_combined_hash();

    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, vector[CAction] &legal_actions, int action_space_size, int num_of_sampled_actions, bool continuous_action_space) except +
        int visit_count, to_play, current_latent_state_index, batch_index
        bool continuous_action_space
        CAction best_action
        float rewards, prior, value_sum
        vector[CNode]* ptr_node_pool;

        void expand(int to_play, int current_latent_state_index, int batch_index, float rewards, vector[float] policy_logits)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float compute_mean_q(int isRoot, float parent_q, float discount_factor)

        int expanded()
        float value()
        vector[vector[float]] get_trajectory()
        vector[int] get_children_distribution()
        CNode* get_child(CAction action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, vector[vector[float]] legal_actions_list, int action_space_size, int num_of_sampled_actions, bool continuous_action_space) except +
        int root_num, action_space_size, num_of_sampled_actions
        bool continuous_action_space
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_noise_weight, const vector[vector[float]] &noises, const vector[float] &rewards, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void prepare_no_noise(const vector[float] &rewards, const vector[vector[float]] &policies, vector[int] to_play_batch)
        void clear()
        vector[vector[vector[float]]] get_trajectories()
        vector[vector[int]] get_distributions()
        vector[vector[vector[float]]] get_sampled_actions()
        vector[float] get_values()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] latent_state_index_in_search_path, latent_state_index_in_batch,  search_lens
        vector[int] virtual_to_play_batchs
        vector[vector[float]] last_actions
        vector[CNode*] nodes

    cdef void cbackpropagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor)
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, vector[float] rewards, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] &to_play_batch)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] &virtual_to_play_batch, bool continuous_action_space)
