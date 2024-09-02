// C++11

#ifndef CNODE_H
#define CNODE_H

#include "../../common_lib/cminimax.h"
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

namespace tree
{
    // sampled related core code
    class CAction
    {
    public:
        std::vector<float> value;
        std::vector<size_t> hash;
        int is_root_action;

        CAction();
        CAction(std::vector<float> value, int is_root_action);
        ~CAction();

        std::vector<size_t> get_hash(void);
        std::size_t get_combined_hash(void);
    };

    class CNode
    {
    public:
        int visit_count, to_play, current_latent_state_index, batch_index, is_reset, action_space_size;
        // sampled related core code
        CAction best_action;
        int num_of_sampled_actions;
        float reward, prior, value_sum;
//        float parent_value_prefix;
        bool continuous_action_space;
        std::vector<int> children_index;
        std::map<size_t, CNode> children;

        std::vector<CAction> legal_actions;

        CNode();
        // sampled related core code
        CNode(float prior, std::vector<CAction> &legal_actions, int action_space_size, int num_of_sampled_actions, bool continuous_action_space);
        ~CNode();

        void expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits);
        void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
        float compute_mean_q(int isRoot, float parent_q, float discount_factor);
        void print_out();

        int expanded();

        float value();

        // sampled related core code
        std::vector<std::vector<float> > get_trajectory();
        std::vector<int> get_children_distribution();
        CNode *get_child(CAction action);
    };

    class CRoots
    {
    public:
        int root_num;
        int num_of_sampled_actions;
        int action_space_size;
        std::vector<CNode> roots;
        std::vector<std::vector<float> > legal_actions_list;
        bool continuous_action_space;

        CRoots();
        CRoots(int root_num, std::vector<std::vector<float> > legal_actions_list, int action_space_size, int num_of_sampled_actions, bool continuous_action_space);
        ~CRoots();

        void prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
        void prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
        void clear();
        // sampled related core code
        std::vector<std::vector<std::vector<float> > > get_trajectories();
        std::vector<std::vector<std::vector<float> > > get_sampled_actions();

        std::vector<std::vector<int> > get_distributions();

        std::vector<float> get_values();
    };

    class CSearchResults
    {
    public:
        int num;
        std::vector<int> latent_state_index_in_search_path, latent_state_index_in_batch, search_lens;
        std::vector<int> virtual_to_play_batchs;
        std::vector<std::vector<float> > last_actions;

        std::vector<CNode *> nodes;
        std::vector<std::vector<CNode *> > search_paths;

        CSearchResults();
        CSearchResults(int num);
        ~CSearchResults();
    };

    //*********************************************************
    void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players);
    void cbackpropagate(std::vector<CNode *> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor);
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch);
    CAction cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, bool continuous_action_space);
    float cucb_score(CNode *parent, CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players, bool continuous_action_space);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, bool continuous_action_space);
}

#endif