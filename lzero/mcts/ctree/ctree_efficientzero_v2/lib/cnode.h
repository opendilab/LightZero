// C++11

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
#include <time.h>
#include <map>

const int DEBUG_MODE = 0;

namespace tree {
    class CNode {
        public:
            int visit_count, to_play, current_latent_state_index, batch_index, best_action, is_reset;
            float value_prefix, prior, value_sum;
            float parent_value_prefix;
            std::vector<int> children_index;
            std::map<int, CNode> children;

            std::vector<int> legal_actions;

            // ========== V2 新增字段 ==========
            std::vector<int> selected_children_idx;      // Sequential Halving 选中的动作
            std::vector<float> estimated_value_lst;      // 多值估计列表（EfficientZero V2 原版风格，未启用）
            float discount;                              // 折扣因子（V2 用）

            CNode();
            CNode(float prior, std::vector<int> &legal_actions);
            ~CNode();

            void expand(int to_play, int current_latent_state_index, int batch_index, float value_prefix, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            float compute_mean_q(int isRoot, float parent_q, float discount_factor);
            void print_out();

            int expanded();

            float value();
            float value_v2();  // EfficientZero V2 原版风格的值估计（基于 estimated_value_lst，未启用）

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);

            // ========== EfficientZero V2 新增方法 ==========
            std::vector<float> get_children_priors();
            std::vector<int> get_children_visits();      // 获取子节点访问次数
            float get_reward();
            float get_qsa(int action);
            float get_v_mix();
            std::vector<float> get_completed_Q(tools::CMinMaxStats &min_max_stats, int to_normalize);
            std::vector<float> get_improved_policy(const std::vector<float> &transformed_completed_Qs);
            int do_equal_visit(int num_simulations);    // Sequential Halving 等量访问策略
    };

    class CRoots{
        public:
            int root_num;
            std::vector<CNode> roots;
            std::vector<std::vector<int> > legal_actions_list;

            CRoots();
            CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list);
            ~CRoots();

            void prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch);
            void clear();
            std::vector<std::vector<int> > get_trajectories();
            std::vector<std::vector<int> > get_distributions();
            std::vector<float> get_values();
            std::vector<std::vector<float> > get_root_policies(tools::CMinMaxStatsList *min_max_stats_lst);
            std::vector<int> get_best_actions();
            CNode* get_root(int index);
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
    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch);
    void cbatch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch, std::vector<int> &no_inference_lst, std::vector<int> &reuse_lst, std::vector<int> &reuse_value_lst);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players);
    int cselect_root_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, int true_action, float reuse_value);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players);
    float carm_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float reuse_value, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players);

    // ========== MuZero/UCB 风格的遍历（备份） ==========
    void cbatch_traverse_ucb(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch);
    void cbatch_traverse_with_reuse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, std::vector<int> &true_action, std::vector<float> &reuse_value);

    // ========== EfficientZero V2 风格的遍历（Sequential Halving 集成） ==========
    void cbatch_traverse(CRoots *roots, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results,
                         int num_simulations, int simulation_idx, const std::vector<std::vector<float>>& gumble_noise,
                         int current_num_top_actions, std::vector<int> &virtual_to_play_batch);

    // ========== EfficientZero V2 Sequential Halving 相关 ==========
    std::vector<float> softmax(const std::vector<float> &logits);
    std::vector<float> get_transformed_completed_Qs(CNode* node, tools::CMinMaxStats &min_max_stats, int final);
    int sequential_halving(CNode* root, const std::vector<float>& gumbel_noise, tools::CMinMaxStats &min_max_stats, int current_phase, int current_num_top_actions);
    std::vector<int> c_batch_sequential_halving(CRoots *roots, const std::vector<std::vector<float>>& gumbel_noises, tools::CMinMaxStatsList *min_max_stats_lst, int current_phase, int current_num_top_actions);

    // 辅助函数
    float max_float(const std::vector<float> &arr);
    float min_float(const std::vector<float> &arr);
    int max_int(const std::vector<int> &arr);
    float sum_float(const std::vector<float> &arr);
    int sum_int(const std::vector<int> &arr);
    int argmax(const std::vector<float> &arr);
}

#endif