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
            std::vector<int> children_index;
            std::map<int, CNode> children;

            std::vector<int> legal_actions;

            // ===== Sequential Halving 根节点特有 =====
            std::vector<int> selected_children_idx;  // 当前阶段的候选动作列表

            CNode();
            CNode(float prior, std::vector<int> &legal_actions);
            ~CNode();

            void expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float compute_mean_q(int isRoot, float parent_q, float discount_factor);
            void print_out();

            int expanded();

            float value();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);

            // ===== Sequential Halving 函数 =====
            int select_root_child_sh(const std::vector<float> &gumble_noise);
            void update_selected_actions(const std::vector<float> &gumble_noise, tools::CMinMaxStats &min_max_stats, int new_num_top_actions);
    };

    class CRoots{
        public:
            int root_num;
            std::vector<CNode> roots;
            std::vector<std::vector<int> > legal_actions_list;

            // ===== Sequential Halving 全局状态 =====
            int use_sequential_halving;              // 是否启用 SH
            int num_simulations;                     // 总 simulation 次数
            int num_top_actions;                     // 初始候选数
            int current_phase;                       // 当前阶段
            int current_num_top_actions;             // 当前阶段的候选数
            int used_visit_num;                      // 已使用的 sim 计数
            int visit_num_for_next_phase;            // 下一阶段转换点
            std::vector<std::vector<float> > stored_gumble_noise;  // 每根一份 Gumbel 噪声

            CRoots();
            CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list);
            ~CRoots();

            void prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void clear();
            std::vector<std::vector<int> > get_trajectories();
            std::vector<std::vector<int> > get_distributions();
            std::vector<float> get_values();

            // ===== Sequential Halving 函数 =====
            void init_sequential_halving(int num_sims, int num_top_acts);
            int ready_for_next_sh_phase();
            void apply_next_sh_phase(tools::CMinMaxStatsList *min_max_stats_lst);
            void set_used_visit_num(int num);

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
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players);
    void cbackpropagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor);
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch);
    void cbatch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch, std::vector<int> &no_inference_lst, std::vector<int> &reuse_lst, std::vector<int> &reuse_value_lst);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players);
    int cselect_root_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, int true_action, float reuse_value);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players);
    float carm_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float reuse_value, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch);
    void cbatch_traverse_with_reuse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, std::vector<int> &true_action, std::vector<float> &reuse_value);
}

#endif