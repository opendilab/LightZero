// C++11

#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <cassert>
#include <climits>

#ifdef _WIN32
#include "..\..\common_lib\utils.cpp"
#else
#include "../../common_lib/utils.cpp"
#endif


namespace tree
{

    CSearchResults::CSearchResults()
    {
        /*
        Overview:
            Initialization of CSearchResults, the default result number is set to 0.
        */
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num)
    {
        /*
        Overview:
            Initialization of CSearchResults with result number.
        */
        this->num = num;
        for (int i = 0; i < num; ++i)
        {
            this->search_paths.push_back(std::vector<CNode *>());
        }
    }

    CSearchResults::~CSearchResults() {}

    //*********************************************************

    CNode::CNode()
    {
        /*
        Overview:
            Initialization of CNode.
        */
        this->prior = 0;
        this->legal_actions = legal_actions;

        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->to_play = 0;
        this->reward = 0.0;

        // ===== Sequential Halving 初始化 =====
        this->selected_children_idx = std::vector<int>();
    }

    CNode::CNode(float prior, std::vector<int> &legal_actions)
    {
        /*
        Overview:
            Initialization of CNode with prior value and legal actions.
        Arguments:
            - prior: the prior value of this node.
            - legal_actions: a vector of legal actions of this node.
        */
        this->prior = prior;
        this->legal_actions = legal_actions;

        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->to_play = 0;
        this->current_latent_state_index = -1;
        this->batch_index = -1;

        // ===== Sequential Halving 初始化 =====
        this->selected_children_idx = std::vector<int>();
    }

    CNode::~CNode() {}

    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play: which player to play the game in the current node.
            - current_latent_state_index: The index of latent state of the leaf node in the search path of the current node.
            - batch_index: The index of latent state of the leaf node in the search path of the current node.
            - reward: the reward of the current node.
            - policy_logits: the logit of the child nodes.
        */
        this->to_play = to_play;
        this->current_latent_state_index = current_latent_state_index;
        this->batch_index = batch_index;
        this->reward = reward;

        int action_num = policy_logits.size();
        if (this->legal_actions.size() == 0)
        {
            for (int i = 0; i < action_num; ++i)
            {
                this->legal_actions.push_back(i);
            }
        }
        float temp_policy;
        float policy_sum = 0.0;

        #ifdef _WIN32
        // 创建动态数组
        float* policy = new float[action_num];
        #else
        float policy[action_num];
        #endif

        float policy_max = FLOAT_MIN;
        for (auto a : this->legal_actions)
        {
            if (policy_max < policy_logits[a])
            {
                policy_max = policy_logits[a];
            }
        }

        for (auto a : this->legal_actions)
        {
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        for (auto a : this->legal_actions)
        {
            prior = policy[a] / policy_sum;
            std::vector<int> tmp_empty;
            this->children[a] = CNode(prior, tmp_empty); // only for muzero/efficient zero, not support alphazero
        }
        
        #ifdef _WIN32
        // 释放数组内存
        delete[] policy;
        #else
        #endif
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises)
    {
        /*
        Overview:
            Add a noise to the prior of the child nodes.
        Arguments:
            - exploration_fraction: the fraction to add noise.
            - noises: the vector of noises added to each child node.
        */
        float noise, prior;
        for (int i = 0; i < this->legal_actions.size(); ++i)
        {
            noise = noises[i];
            CNode *child = this->get_child(this->legal_actions[i]);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::compute_mean_q(int isRoot, float parent_q, float discount_factor)
    {
        /*
        Overview:
            Compute the mean q value of the current node.
        Arguments:
            - isRoot: whether the current node is a root node.
            - parent_q: the q value of the parent node.
            - discount_factor: the discount_factor of reward.
        */
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        for (auto a : this->legal_actions)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
                float true_reward = child->reward;
                float qsa = true_reward + discount_factor * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if (isRoot && total_visits > 0)
        {
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else
        {
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    void CNode::print_out()
    {
        return;
    }

    int CNode::expanded()
    {
        /*
        Overview:
            Return whether the current node is expanded.
        */
        return this->children.size() > 0;
    }

    float CNode::value()
    {
        /*
        Overview:
            Return the real value of the current tree.
        */
        float true_value = 0.0;
        if (this->visit_count == 0)
        {
            return true_value;
        }
        else
        {
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<int> CNode::get_trajectory()
    {
        /*
        Overview:
            Find the current best trajectory starts from the current node.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from this node.
        */
        std::vector<int> traj;

        CNode *node = this;
        int best_action = node->best_action;
        while (best_action >= 0)
        {
            traj.push_back(best_action);

            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> CNode::get_children_distribution()
    {
        /*
        Overview:
            Get the distribution of child nodes in the format of visit_count.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<int> distribution;
        if (this->expanded())
        {
            for (auto a : this->legal_actions)
            {
                CNode *child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode *CNode::get_child(int action)
    {
        /*
        Overview:
            Get the child node corresponding to the input action.
        Arguments:
            - action: the action to get child.
        */
        return &(this->children[action]);
    }

    int CNode::select_root_child_sh(const std::vector<float> &gumble_noise)
    {
        /*
        Overview:
            Select a child action for root node using Sequential Halving.
            Select the least visited action among the current candidates.
        Arguments:
            - gumble_noise: Gumbel noise array
        Returns:
            - The action index to select (least visited among candidates)
        */
        int selected_action = -1;
        int min_visit_count = INT_MAX;

        // If candidates not yet initialized, initialize all legal actions as candidates
        if (this->selected_children_idx.empty()) {
            for (int action : this->legal_actions) {
                this->selected_children_idx.push_back(action);
            }
        }

        // Select the least visited action among candidates (equal visit strategy)
        for (int action : this->selected_children_idx) {
            CNode *child = this->get_child(action);
            if (child != nullptr && child->visit_count < min_visit_count) {
                min_visit_count = child->visit_count;
                selected_action = action;
            }
        }

        // If no candidate found, select first legal action
        if (selected_action == -1 && !this->legal_actions.empty()) {
            selected_action = this->legal_actions[0];
        }

        return selected_action;
    }

    void CNode::update_selected_actions(const std::vector<float> &gumble_noise, tools::CMinMaxStats &min_max_stats, int new_num_top_actions)
    {
        /*
        Overview:
            Prune candidate actions in Sequential Halving: keep only top-k actions based on Gumbel-augmented Q-values.
        Arguments:
            - gumble_noise: Gumbel noise array
            - min_max_stats: min-max statistics for value normalization
            - new_num_top_actions: number of actions to keep in next phase
        */
        if (this->selected_children_idx.empty()) {
            return;
        }

        // Calculate Gumbel-augmented scores for each candidate
        std::vector<std::pair<float, int>> scored_actions;  // (score, action)

        for (int action : this->selected_children_idx) {
            CNode *child = this->get_child(action);
            if (child == nullptr) continue;

            // Calculate score = gumbel[action] + prior[action] + normalized_q[action]
            float q_value = child->value();
            float normalized_q = min_max_stats.normalize(q_value);
            float gumbel_score = gumble_noise[action] + child->prior + normalized_q;

            scored_actions.push_back({gumbel_score, action});
        }

        // Sort by score in descending order and keep top-k
        std::sort(scored_actions.begin(), scored_actions.end(),
                  [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                      return a.first > b.first;
                  });

        // Update selected actions to keep only top-k
        this->selected_children_idx.clear();
        for (int i = 0; i < std::min((int)scored_actions.size(), new_num_top_actions); ++i) {
            this->selected_children_idx.push_back(scored_actions[i].second);
        }
    }

    //*********************************************************

    CRoots::CRoots()
    {
        /*
        Overview:
            The initialization of CRoots.
        */
        this->root_num = 0;
    }

    CRoots::CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list)
    {
        /*
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num: the number of the current root.
            - legal_action_list: the vector of the legal action of this root.
        */
        this->root_num = root_num;
        this->legal_actions_list = legal_actions_list;

        for (int i = 0; i < root_num; ++i)
        {
            this->roots.push_back(CNode(0, this->legal_actions_list[i]));
        }
    }

    CRoots::~CRoots() {}

    void CRoots::prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i]);
            this->roots[i].add_exploration_noise(root_noise_weight, noises[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots without noise.
        Arguments:
            - rewards: the vector of rewards of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear()
    {
        /*
        Overview:
            Clear the roots vector.
        */
        this->roots.clear();
    }

    std::vector<std::vector<int> > CRoots::get_trajectories()
    {
        /*
        Overview:
            Find the current best trajectory starts from each root.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from each root.
        */
        std::vector<std::vector<int> > trajs;
        trajs.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions()
    {
        /*
        Overview:
            Get the children distribution of each root.
        Returns:
            - distribution: a vector of distribution of child nodes in the format of visit count (i.e. [1,3,0,2,5]).
        */
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values()
    {
        /*
        Overview:
            Return the real value of each root.
        */
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    // ===== Sequential Halving 初始化 =====
    void CRoots::init_sequential_halving(int num_sims, int num_top_acts)
    {
        /*
        Overview:
            Initialize Sequential Halving parameters for the roots.
        Arguments:
            - num_sims: total number of simulations (e.g., 50)
            - num_top_acts: initial number of top actions to consider (e.g., 16)

        Design:
            Precompute the transition point for each phase using the official SH formula:
            visit_num_for_next_phase = max(floor(n / (log2(m) * current_m)), 1) * current_m
            where n = num_simulations, m = num_top_actions
        */
        this->use_sequential_halving = 1;
        this->num_simulations = num_sims;
        this->num_top_actions = num_top_acts;
        this->current_phase = 0;
        this->current_num_top_actions = num_top_acts;
        this->used_visit_num = 0;

        // Calculate log2(num_top_acts) - number of phases
        int log2_m = 0;
        int temp = num_top_acts;
        while (temp > 1) {
            temp /= 2;
            log2_m++;
        }
        if (log2_m == 0) log2_m = 1;  // Ensure at least 1 phase

        // Precompute visit count for next phase transition
        // Formula: max(floor(n / (log2(m) * current_m)), 1) * current_m
        this->visit_num_for_next_phase = std::max(
            (int)std::floor((float)num_sims / (log2_m * this->current_num_top_actions)),
            1
        ) * this->current_num_top_actions;

        // Initialize stored Gumbel noise with default value 0.0
        // Will be updated by batch_traverse when actual Gumbel noise is passed
        this->stored_gumble_noise.clear();
        for (int i = 0; i < this->root_num; ++i) {
            // Assume legal_actions_list[i].size() is the number of actions for this root
            std::vector<float> zero_noise(this->legal_actions_list[i].size(), 0.0);
            this->stored_gumble_noise.push_back(zero_noise);
        }
    }

    int CRoots::ready_for_next_sh_phase()
    {
        /*
        Overview:
            Check if ready to transition to the next Sequential Halving phase.
        Returns:
            - 1 if ready to transition, 0 otherwise

        Logic:
            - Check if used_visit_num has reached visit_num_for_next_phase
            - Also ensure current_num_top_actions > 1 (can still be halved)
        */
        if (this->use_sequential_halving == 0) return 0;

        if (this->used_visit_num >= this->visit_num_for_next_phase &&
            this->current_num_top_actions > 1) {
            return 1;
        }
        return 0;
    }

    void CRoots::apply_next_sh_phase(tools::CMinMaxStatsList *min_max_stats_lst)
    {
        /*
        Overview:
            Apply the next Sequential Halving phase: reduce candidate actions and update visit counts.
        Arguments:
            - min_max_stats_lst: the min-max statistics list for normalization

        Steps:
            1. Halve the number of top actions: current_num_top_actions /= 2
            2. Move to next phase: current_phase++
            3. Reset used_visit_num for next phase counting
            4. Recalculate visit_num_for_next_phase with new current_num_top_actions
            5. Call update_selected_actions for each root to prune candidates
        */
        if (this->use_sequential_halving == 0) return;

        printf("\n[SH阶段转换] 进入新阶段\n");
        printf("  转换前: 阶段=%d 候选数=%d\n", this->current_phase, this->current_num_top_actions);
        fflush(stdout);

        // Step 1: Halve the number of top actions
        this->current_num_top_actions = std::max(this->current_num_top_actions / 2, 1);
        this->current_phase++;

        // Step 2: Reset used visit count for next phase
        this->used_visit_num = 0;

        // Step 3: Recalculate visit count for next phase transition
        if (this->current_num_top_actions > 1) {
            int log2_m = 0;
            int temp = this->num_top_actions;
            while (temp > 1) {
                temp /= 2;
                log2_m++;
            }
            if (log2_m == 0) log2_m = 1;

            this->visit_num_for_next_phase = std::max(
                (int)std::floor((float)this->num_simulations / (log2_m * this->current_num_top_actions)),
                1
            ) * this->current_num_top_actions;
        }

        // Step 4: Update selected actions for each root based on current phase
        for (int i = 0; i < this->root_num; ++i) {
            if (this->stored_gumble_noise.size() > (size_t)i) {
                this->roots[i].update_selected_actions(
                    this->stored_gumble_noise[i],
                    min_max_stats_lst->stats_lst[i],
                    this->current_num_top_actions
                );
            }
        }

        printf("  转换后: 阶段=%d 候选数=%d 下次转换点=%d\n",
               this->current_phase, this->current_num_top_actions, this->visit_num_for_next_phase);
        fflush(stdout);
    }

    void CRoots::set_used_visit_num(int num)
    {
        /*
        Overview:
            Set the used visit count for Sequential Halving phase transition checking.
        Arguments:
            - num: the number of visits to set
        */
        this->used_visit_num = num;
    }

    void cbackpropagate(std::vector<CNode *> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount_factor)
    {
        /*
        Overview:
            Update the value sum and visit count of nodes along the search path.
        Arguments:
            - search_path: a vector of nodes on the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - to_play: which player to play the game in the current node.
            - value: the value to propagate along the search path.
            - discount_factor: the discount factor of reward.
        */
        assert(to_play == -1 || to_play == 1 || to_play == 2);
        if (to_play == -1)
        {           
            // for play-with-bot-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float true_reward = node->reward;

                min_max_stats.update(true_reward + discount_factor * node->value());

                bootstrap_value = true_reward + discount_factor * bootstrap_value;
            }
        }
        else
        {
            // for self-play-mode
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                if (node->to_play == to_play)
                    node->value_sum += bootstrap_value;
                else
                    node->value_sum += -bootstrap_value;
                node->visit_count += 1;

                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                //                float true_reward = node->value_prefix - parent_value_prefix;
                float true_reward = node->reward;

                // TODO(pu): why in muzero-general is - node.value
                min_max_stats.update(true_reward + discount_factor * -node->value());

                if (node->to_play == to_play)
                    bootstrap_value = -true_reward + discount_factor * bootstrap_value;
                else
                    bootstrap_value = true_reward + discount_factor * bootstrap_value;
            }
        }
    }

    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - current_latent_state_index: The index of latent state of the leaf node in the search path.
            - discount_factor: the discount factor of reward.
            - rewards: the rewards of nodes along the search path.
            - values: the values to propagate along the search path.
            - policies: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - to_play_batch: the batch of which player is playing on this node.
        */
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, i, rewards[i], policies[i]);
            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount_factor);
        }
    }

    void cbatch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch, std::vector<int> &no_inference_lst, std::vector<int> &reuse_lst, std::vector<float> &reuse_value_lst)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.  Details are similar to cbatch_backpropagate, but with reuse value.
            Please refer to https://arxiv.org/abs/2404.16364 for more details.
        Arguments:
            - current_latent_state_index: The index of latent state of the leaf node in the search path.
            - discount_factor: the discount factor of reward.
            - value_prefixs: the value prefixs of nodes along the search path.
            - values: the values to propagate along the search path.
            - policies: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - to_play_batch: the batch of which player is playing on this node.
            - no_inference_lst: the list of the nodes which does not need to expand.
            - reuse_lst: the list of the nodes which should use reuse-value to backpropagate.
            - reuse_value_lst: the list of the reuse-value.
        */
        int count_a = 0;
        int count_b = 0;
        int count_c = 0;
        float value_propagate = 0;
        for (int i = 0; i < results.num; ++i)
        {
            if (i == no_inference_lst[count_a])
            {
                count_a = count_a + 1;
                value_propagate = reuse_value_lst[i];
            }
            else
            {
                results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, count_b, value_prefixs[count_b], policies[count_b]);
                if (i == reuse_lst[count_c])
                {
                    value_propagate = reuse_value_lst[i];
                    count_c = count_c + 1;
                }
                else
                {
                    value_propagate = values[count_b];
                }
                count_b = count_b + 1;
            }

            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], value_propagate, discount_factor);
        }
    }

    int cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players)
    {
        /*
        Overview:
            Select the child node of the roots according to ucb scores.
        Arguments:
            - root: the roots to select the child node.
            - min_max_stats: a tool used to min-max normalize the score.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - mean_q: the mean q value of the parent node.
            - players: the number of players.
        Returns:
            - action: the action to select.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        for (auto a : root->legal_actions)
        {

            CNode *child = root->get_child(a);
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->visit_count - 1, pb_c_base, pb_c_init, discount_factor, players);

            if (max_score < temp_score)
            {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(a);
            }
        }

        int action = 0;
        if (max_index_lst.size() > 0)
        {
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    int cselect_root_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, int true_action, float reuse_value)
    {
        /*
        Overview:
            Select the child node of the roots according to ucb scores.
        Arguments:
            - root: the roots to select the child node.
            - min_max_stats: a tool used to min-max normalize the score.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - discount_factor: the discount factor of reward.
            - mean_q: the mean q value of the parent node.
            - players: the number of players.
            - true_action: the action chosen in the trajectory.
            - reuse_value: the value obtained from the search of the next state in the trajectory.
        Returns:
            - action: the action to select.
        */
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        for (auto a : root->legal_actions)
        {

            CNode *child = root->get_child(a);
            float temp_score = 0.0;
            if (a == true_action)
            {
                temp_score = carm_score(child, min_max_stats, mean_q, reuse_value, root->visit_count - 1, pb_c_base, pb_c_init, discount_factor, players);
            }
            else
            {
                temp_score = cucb_score(child, min_max_stats, mean_q, root->visit_count - 1, pb_c_base, pb_c_init, discount_factor, players);
            }

            if (max_score < temp_score)
            {
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if (temp_score >= max_score - epsilon)
            {
                max_index_lst.push_back(a);
            }
        }

        int action = 0;
        if (max_index_lst.size() > 0)
        {
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - mean_q: the mean q value of the parent node.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
        Returns:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->reward;
            if (players == 1)
                value_score = true_reward + discount_factor * child->value();
            else if (players == 2)
                value_score = true_reward + discount_factor * (-child->value());
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }


    float carm_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float reuse_value, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - mean_q: the mean q value of the parent node.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
        Returns:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->reward;
            if (players == 1)
                value_score = true_reward + discount_factor * reuse_value;
            else if (players == 2)
                value_score = true_reward + discount_factor * (-reuse_value);
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;
        float ucb_value = 0.0;
        if (child->visit_count == 0)
        {
            ucb_value = prior_score + value_score;
        }
        else
        {
            ucb_value = value_score;
        }
        return ucb_value;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch)
    {
        /*
        Overview:
            Search node path from the roots.
        Arguments:
            - roots: the roots that search from.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - min_max_stats: a tool used to min-max normalize the score.
            - results: the search results.
            - virtual_to_play_batch: the batch of which player is playing on this node.
        */
        // set seed
        get_time_and_set_rand_seed();

        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end()); // 0 or 2
        if (largest_element == -1)
            players = 1;
        else
            players = 2;

        for (int i = 0; i < results.num; ++i)
        {
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                parent_q = mean_q;

                int action;
                if (is_root == 1 && roots->use_sequential_halving) {
                    action = node->select_root_child_sh(roots->stored_gumble_noise[i]);
                    // 调试日志：显示根节点选择
                    printf("[SH选择] 环境%d 当前阶段=%d 候选数=%d 选择动作=%d (访问次数=%d)\n",
                           i, roots->current_phase, roots->current_num_top_actions,
                           action, node->get_child(action)->visit_count);
                    fflush(stdout);
                } else {
                    action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players);
                    // 调试日志：显示深节点选择
                    if (is_root == 0 && search_len < 3) {  // 只显示前3层
                        printf("[UCB选择] 环境%d 深度=%d 选择动作=%d (访问次数=%d 价值=%.4f)\n",
                               i, search_len, action, node->get_child(action)->visit_count,
                               node->get_child(action)->value());
                        fflush(stdout);
                    }
                }

                is_root = 0;
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                        virtual_to_play_batch[i] = 2;
                    else
                        virtual_to_play_batch[i] = 1;
                }

                node->best_action = action;
                // next
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.latent_state_index_in_search_path.push_back(parent->current_latent_state_index);
            results.latent_state_index_in_batch.push_back(parent->batch_index);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
            results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);
        }
    }


    void cbatch_traverse_with_reuse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, std::vector<int> &true_action, std::vector<float> &reuse_value)
    {
        /*
        Overview:
            Search node path from the roots. Details are similar to cbatch_traverse, but with reuse value.
            Please refer to https://arxiv.org/abs/2404.16364 for more details.
        Arguments:
            - roots: the roots that search from.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - min_max_stats: a tool used to min-max normalize the score.
            - results: the search results.
            - virtual_to_play_batch: the batch of which player is playing on this node.
            - true_action: the action chosen in the trajectory.
            - reuse_value: the value obtained from the search of the next state in the trajectory.
        */
        // set seed
        get_time_and_set_rand_seed();

        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end()); // 0 or 2
        if (largest_element == -1)
            players = 1;
        else
            players = 2;

        for (int i = 0; i < results.num; ++i)
        {
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                parent_q = mean_q;

                int action = 0;
                if (is_root)
                {
                    action = cselect_root_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players, true_action[i], reuse_value[i]);
                }
                else
                {
                    action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players);
                }
                
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                        virtual_to_play_batch[i] = 2;
                    else
                        virtual_to_play_batch[i] = 1;
                }

                node->best_action = action;
                // next
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;

                if(is_root && action == true_action[i])
                {
                    break;
                }

                is_root = 0;

            }

            if (node->expanded())
            {
                results.latent_state_index_in_search_path.push_back(-1);
                results.latent_state_index_in_batch.push_back(i);

                results.last_actions.push_back(last_action);
                results.search_lens.push_back(search_len);
                results.nodes.push_back(node);
                results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);
            }
            else
            {
                CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

                results.latent_state_index_in_search_path.push_back(parent->current_latent_state_index);
                results.latent_state_index_in_batch.push_back(parent->batch_index);

                results.last_actions.push_back(last_action);
                results.search_lens.push_back(search_len);
                results.nodes.push_back(node);
                results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);
            }
        }
    }

}