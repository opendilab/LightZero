#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
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

    class CAction{
        public:
            std::vector<float> value;
            std::vector<size_t> hash;
            int is_root_action;

            CAction();
            CAction(std::vector<float> value, int is_root_action);
            ~CAction();

//            int hash(std::vector<float> value);
//            std::hash<int> get_hash();

            std::vector<size_t> get_hash( void );
            std::size_t get_combined_hash( void );


    };

    class CNode {
        public:
            int visit_count, to_play, hidden_state_index_x, hidden_state_index_y, is_reset, action_space_size;
            // sampled related
            CAction best_action;
            int num_of_sampled_actions;
            float value_prefix, prior, value_sum;
            float parent_value_prefix;
            std::vector<int> children_index;
            // std::vector<CNode>* ptr_node_pool;
//            std::map<int, std::vector<float> > policy;
//            std::vector<int> legal_actions;
//            std::map<int, float> policy;
//            std::map<CAction, CNode> children;
            std::map<size_t, CNode> children;

            std::vector<CAction> legal_actions;

            CNode();
            CNode(float prior, std::vector<CAction> &legal_actions, int action_space_size, int num_of_sampled_actions);
            ~CNode();

            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float get_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();

            std::vector<std::vector<float> > get_trajectory();
//            std::vector<CAction> get_trajectory();
            std::vector<int> get_children_distribution();
//            CNode* get_child(int action);
            CNode* get_child(CAction action);

    };

    class CRoots{
        public:
//            int root_num, pool_size;
            int root_num;
            int num_of_sampled_actions;
            int action_space_size;
            std::vector<CNode> roots;
            // std::vector<std::vector<CNode> > node_pools;
            std::vector<std::vector<std::vector<float> > > legal_actions_list;
//            std::vector<std::vector<CAction> > legal_actions_list;


            CRoots();
            CRoots(int root_num, std::vector<std::vector<std::vector<float> > > legal_actions_list, int action_space_size, int num_of_sampled_actions);
//            CRoots(int root_num, std::vector<std::vector<CAction> > &legal_actions_list, int action_space_size, int num_of_sampled_actions);
            ~CRoots();

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void clear();
            std::vector<std::vector<std::vector<float> > > get_trajectories();
//            std::vector<std::vector<CAction> >* get_trajectories();

            std::vector<std::vector<int> > get_distributions();
            std::vector<float> get_values();

    };

    class CSearchResults{
        public:
            int num;
//            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, search_lens;
            std::vector<std::vector<int> > virtual_to_play_batch;

//            std::vector<CAction> last_actions;
            std::vector<std::vector<float> > last_actions;


            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*> > search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };


    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount, int players);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst, std::vector<int> &to_play_batch);
//    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players);
    CAction cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players);

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount, int players);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch);
}

#endif