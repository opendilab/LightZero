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
#include <time.h>
#include <map>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, current_latent_state_index, batch_index, best_action;
            float reward, prior, value_sum;
            bool is_chance;
            int chance_space_size;
            std::vector<int> children_index;
            std::map<int, CNode> children;

            std::vector<int> legal_actions;

            CNode();
            CNode(float prior, std::vector<int> &legal_actions, bool is_chance = false, int chance_space_size = 2);
            ~CNode();

            void expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits, bool is_chance);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float compute_mean_q(int isRoot, float parent_q, float discount_factor);
            void print_out();

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num;
            std::vector<CNode> roots;
            std::vector<std::vector<int> > legal_actions_list;
            int chance_space_size;

            CRoots();
            CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list, int chance_space_size);
            ~CRoots();

            void prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void clear();
            std::vector<std::vector<int> > get_trajectories();
            std::vector<std::vector<int> > get_distributions();
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, search_lens;
            std::vector<int> virtual_to_play_batchs;
            std::vector<CNode*> nodes;
            std::vector<bool> leaf_node_is_chance;
            std::vector<std::vector<CNode*> > search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players);
    void cbackpropagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor);
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch, std::vector<int> & is_chance_list, std::vector<int> &leaf_idx_list);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch);
}

#endif