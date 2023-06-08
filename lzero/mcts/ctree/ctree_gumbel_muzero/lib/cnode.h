// C++11

#ifndef CNODE_H
#define CNODE_H

#include "./../common_lib/cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>
#include <map>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, current_latent_state_index, batch_index, best_action;
            float reward, prior, value_sum, raw_value, gumbel_scale, gumbel_rng;
            std::vector<int> children_index;
            std::map<int, CNode> children;

            std::vector<int> legal_actions;
            std::vector<float> gumbel;

            CNode();
            CNode(float prior, std::vector<int> &legal_actions);
            ~CNode();

            void expand(int to_play, int current_latent_state_index, int batch_index, float reward, float value, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            std::vector<float> get_q(float discount);
            float compute_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            std::vector<float> get_children_value(float discount_factor, int action_space_size);
            std::vector<float> get_policy(float discount, int action_space_size);
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num;
            std::vector<CNode> roots;
            std::vector<std::vector<int> > legal_actions_list;

            CRoots();
            CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list);
            ~CRoots();

            void prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void prepare_no_noise(const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void clear();
            std::vector<std::vector<int> > get_trajectories();
            std::vector<std::vector<int> > get_distributions();
            std::vector<std::vector<float> > get_children_values(float discount, int action_space_size);
            std::vector<std::vector<float> > get_policies(float discount, int action_space_size);
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, search_lens;
            std::vector<int> virtual_to_play_batchs;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*> > search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount, int players);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cbatch_back_propagate(int current_latent_state_index, float discount, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch);
    int cselect_root_child(CNode* root, float discount, int num_simulations, int max_num_considered_actions);
    int cselect_interior_child(CNode* root, float discount);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount, int players);
    void cbatch_traverse(CRoots *roots, int num_simulations, int max_num_considered_actions, float discount, CSearchResults &results, std::vector<int> &virtual_to_play_batch);
    void csoftmax(std::vector<float> &input, int input_len);
    float compute_mixed_value(float raw_value, std::vector<float> q_values, std::vector<int> &child_visit, std::vector<float> &child_prior);
    void rescale_qvalues(std::vector<float> &value, float epsilon);
    std::vector<float> qtransform_completed_by_mix_value(CNode *root, std::vector<int> & child_visit, \
        std::vector<float> & child_prior, float discount= 0.99, float maxvisit_init = 50.0, float value_scale = 0.1, \
        bool rescale_values = true, float epsilon = 1e-8);
    std::vector<int> get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations);
    std::vector<std::vector<int> > get_table_of_considered_visits(int max_num_considered_actions, int num_simulations);
    std::vector<float> score_considered(int considered_visit, std::vector<float> gumbel, std::vector<float> logits, std::vector<float> normalized_qvalues, std::vector<int> visit_counts);
    std::vector<float> generate_gumbel(float gumbel_scale, float gumbel_rng, int shape);
}

#endif
