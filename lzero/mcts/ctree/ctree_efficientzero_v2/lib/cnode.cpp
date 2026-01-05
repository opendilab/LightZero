// C++11

#include <iostream>
#include "cnode.h"
#include "cminimax.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <cassert>

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

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
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

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->best_action = -1;
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
        this->current_latent_state_index = -1;
        this->batch_index = -1;
    }

    CNode::~CNode() {}

    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float value_prefix, const std::vector<float> &policy_logits)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play: which player to play the game in the current node.
            - current_latent_state_index: the x/first index of hidden state vector of the current node, i.e. the search depth.
            - batch_index: the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - value_prefix: the value prefix of the current node.
            - policy_logits: the policy logit of the child nodes.
        */
        this->to_play = to_play;
        this->current_latent_state_index = current_latent_state_index;
        this->batch_index = batch_index;
        this->value_prefix = value_prefix;

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
        float parent_value_prefix = this->value_prefix;
        for (auto a : this->legal_actions)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
                float true_reward = child->value_prefix - parent_value_prefix;
                if (this->is_reset == 1)
                {
                    true_reward = child->value_prefix;
                }
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
            Return the estimated value of the current tree.
            Current implementation: uses value_sum / visit_count (traditional MCTS style).
        Returns:
            The mean value of all backpropagations through this node.
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

    float CNode::value_v2()
    {
        /*
        Overview:
            EfficientZero V2 original style value estimation using estimated_value_lst.
            This method is NOT USED in current implementation, kept for reference.

        Difference from value():
            - value():    uses value_sum / visit_count  (O(1) memory, cumulative sum)
            - value_v2(): uses sum(estimated_value_lst) / len(estimated_value_lst)  (O(n) memory, keeps history)

        Mathematical equivalence:
            Both methods produce the same average value.

        Why keep estimated_value_lst?
            1. Reanalyze: Can re-evaluate nodes with new network
            2. Uncertainty estimation: Can compute std(estimated_value_lst)
            3. Value distribution: Can analyze value distribution
            4. Advanced features: Ensemble learning, distributed training

        Current status:
            UNUSED - The estimated_value_lst is initialized but not populated.
            To enable: uncomment the push_back lines in cbackpropagate().
        */
        if (this->estimated_value_lst.empty())
        {
            return 0.0;
        }

        float sum = 0.0;
        for (float v : this->estimated_value_lst)
        {
            sum += v;
        }
        return sum / this->estimated_value_lst.size();
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

    void CRoots::prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots and add noises.
        Arguments:
            - root_noise_weight: the exploration fraction of roots
            - noises: the vector of noise add to the roots.
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].add_exploration_noise(root_noise_weight, noises[i]);
            this->roots[i].visit_count += 1;

            // Initialize selected_children_idx with all legal actions for Sequential Halving
            this->roots[i].selected_children_idx.clear();
            for (int action : this->roots[i].legal_actions) {
                this->roots[i].selected_children_idx.push_back(action);
            }
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the roots without noise.
        Arguments:
            - value_prefixs: the vector of value prefixs of each root.
            - policies: the vector of policy logits of each root.
            - to_play_batch: the vector of the player side of each root.
        */
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].visit_count += 1;

            // Initialize selected_children_idx with all legal actions for Sequential Halving
            this->roots[i].selected_children_idx.clear();
            for (int action : this->roots[i].legal_actions) {
                this->roots[i].selected_children_idx.push_back(action);
            }
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
            Return the estimated value of each root.
        */
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    std::vector<std::vector<float>> CRoots::get_root_policies(tools::CMinMaxStatsList *min_max_stats_lst)
    {
        /*
        Overview:
            Get improved policies for each root based on MCTS search results.
            The improved policy is computed as softmax(prior + transformed_Q).
        Arguments:
            - min_max_stats_lst: min-max statistics for Q value normalization.
        Returns:
            - policies: vector of policy distributions for each root.
        */
        std::vector<std::vector<float>> policies;
        policies.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            // Get transformed completed Q values (with Sigma transform)
            std::vector<float> transformed_completed_Qs =
                get_transformed_completed_Qs(&(this->roots[i]), min_max_stats_lst->stats_lst[i], 0);

            // Get improved policy based on Q values: softmax(prior + transformed_Q)
            std::vector<float> improved_policy =
                this->roots[i].get_improved_policy(transformed_completed_Qs);

            policies.push_back(improved_policy);
        }
        return policies;
    }

    std::vector<int> CRoots::get_best_actions()
    {
        /*
        Overview:
            Get the best action for each root after Sequential Halving.
            The best action is the first element in selected_children_idx,
            which has the highest score (Gumbel noise + prior + transformed Q).
        Returns:
            - best_actions: vector of best action indices for each root.
        */
        std::vector<int> best_actions(this->root_num, -1);

        for (int i = 0; i < this->root_num; ++i)
        {
            // selected_children_idx[0] is the action with highest score after Sequential Halving
            best_actions[i] = this->roots[i].selected_children_idx[0];
        }
        return best_actions;
    }

    //*********************************************************
    //
    void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats, float discount_factor, int players)
    {
        /*
        Overview:
            Update the q value of the root and its child nodes.
        Arguments:
            - root: the root that update q value from.
            - min_max_stats: a tool used to min-max normalize the q value.
            - discount_factor: the discount factor of reward.
            - players: the number of players.
        */
        std::stack<CNode *> node_stack;
        node_stack.push(root);
        float parent_value_prefix = 0.0;
        int is_reset = 0;
        while (node_stack.size() > 0)
        {
            CNode *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                // true_reward = node.value_prefix - (- parent_value_prefix)
                float true_reward = node->value_prefix - node->parent_value_prefix;

                if (is_reset == 1)
                {
                    true_reward = node->value_prefix;
                }
                float qsa;
                if (players == 1)
                {
                    qsa = true_reward + discount_factor * node->value();
                }
                else if (players == 2)
                {
                    // TODO(pu): why only the last reward multiply the discount_factor?
                    qsa = true_reward + discount_factor * (-1) * node->value();
                }

                min_max_stats.update(qsa);
            }

            for (auto a : node->legal_actions)
            {
                CNode *child = node->get_child(a);
                if (child->expanded())
                {
                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }

            is_reset = node->is_reset;
        }
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

                // ========== Current Implementation: value_sum (Traditional MCTS) ==========
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                // ========== Alternative Implementation: estimated_value_lst (EfficientZero V2 Original) ==========
                // Uncomment the line below to use EfficientZero V2 original style:
                // (node->estimated_value_lst).push_back(bootstrap_value);
                // node->visit_count += 1;
                //
                // Then in value calculation, use:
                //   node->value_v2() instead of node->value()
                //
                // Difference:
                //   - value_sum:           O(1) memory, stores cumulative sum
                //   - estimated_value_lst: O(n) memory, stores all values (enables reanalyze, uncertainty estimation)
                //   - Result: Both produce the same average value mathematically
                // ============================================================================

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                float true_reward = node->value_prefix - parent_value_prefix;
                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset
                    true_reward = node->value_prefix;
                }

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

                // ========== Current Implementation: value_sum (Traditional MCTS) ==========
                if (node->to_play == to_play)
                {
                    node->value_sum += bootstrap_value;
                }
                else
                {
                    node->value_sum += -bootstrap_value;
                }
                node->visit_count += 1;

                // ========== Alternative Implementation: estimated_value_lst (EfficientZero V2 Original) ==========
                // For self-play mode with estimated_value_lst:
                // if (node->to_play == to_play)
                // {
                //     (node->estimated_value_lst).push_back(bootstrap_value);
                // }
                // else
                // {
                //     (node->estimated_value_lst).push_back(-bootstrap_value);
                // }
                // node->visit_count += 1;
                // ============================================================================

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if (i >= 1)
                {
                    CNode *parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                float true_reward = node->value_prefix - parent_value_prefix;

                min_max_stats.update(true_reward + discount_factor * node->value());

                if (is_reset == 1)
                {
                    // parent is reset
                    true_reward = node->value_prefix;
                }
                if (node->to_play == to_play)
                {
                    bootstrap_value = -true_reward + discount_factor * bootstrap_value;
                }
                else
                {
                    bootstrap_value = true_reward + discount_factor * bootstrap_value;
                }
            }
        }
    }

    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
        Arguments:
            - current_latent_state_index: The index of latent state of the leaf node in the search path.
            - discount_factor: the discount factor of reward.
            - value_prefixs: the value prefixs of nodes along the search path.
            - values: the values to propagate along the search path.
            - policies: the policy logits of nodes along the search path.
            - min_max_stats: a tool used to min-max normalize the q value.
            - results: the search results.
            - is_reset_list: the vector of is_reset nodes along the search path, where is_reset represents for whether the parent value prefix needs to be reset.
            - to_play_batch: the batch of which player is playing on this node.
        */
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, i, value_prefixs[i], policies[i]);
            // reset
            results.nodes[i]->is_reset = is_reset_list[i];

            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount_factor);
        }
    }

    void cbatch_backpropagate_with_reuse(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_list, std::vector<int> &to_play_batch, std::vector<int> &no_inference_lst, std::vector<int> &reuse_lst, std::vector<float> &reuse_value_lst)
    {
        /*
        Overview:
            Expand the nodes along the search path and update the infos.
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
            results.nodes[i]->is_reset = is_reset_list[i];
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
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount_factor, players);

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
            - disount_factor: the discount factor of reward.
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
                temp_score = carm_score(child, min_max_stats, mean_q, root->is_reset, reuse_value, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount_factor, players);
            }
            else
            {
                temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount_factor, players);
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
        // printf("select root child ends");
        return action;
    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - is_reset: whether the value prefix needs to be reset.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - parent_value_prefix: the value prefix of parent node.
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
            float true_reward = child->value_prefix - parent_value_prefix;
            if (is_reset == 1)
            {
                true_reward = child->value_prefix;
            }

            if (players == 1)
            {
                value_score = true_reward + discount_factor * child->value();
            }
            else if (players == 2)
            {
                value_score = true_reward + discount_factor * (-child->value());
            }
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
        {
            value_score = 0;
        }
        else if (value_score > 1)
        {
            value_score = 1;
        }

        return prior_score + value_score; // ucb_value
    }

    float carm_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float reuse_value, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount_factor, int players)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - is_reset: whether the value prefix needs to be reset.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - parent_value_prefix: the value prefix of parent node.
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
            float true_reward = child->value_prefix - parent_value_prefix;
            if (is_reset == 1)
            {
                true_reward = child->value_prefix;
            }

            if (players == 1)
            {
                value_score = true_reward + discount_factor * reuse_value;
            }
            else if (players == 2)
            {
                value_score = true_reward + discount_factor * (-reuse_value);
            }
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
        {
            value_score = 0;
        }
        else if (value_score > 1)
        {
            value_score = 1;
        }

        float ucb_value = 0.0;
        if (child->visit_count == 0)
        {
            ucb_value = prior_score + value_score;
        }
        else
        {
            ucb_value = value_score;
        }
        // printf("carmscore ends");
        return ucb_value;
    }

    /**
     * =====================================================================
     * 【EfficientZero V2】动作选择函数
     * =====================================================================
     * 根据当前节点类型和搜索进度采用不同的动作选择策略：
     * 1. 根节点：Sequential Halving + 等量访问（Gumbel-Max 采样初始化）
     * 2. 非根节点：改进策略 + 访问计数平衡
     *
     * 参数：
     *   - node: 当前树节点（可能是根节点或内部节点）
     *   - min_max_stats: 最小最大值统计，用于Q值归一化
     *   - num_simulations: 总模拟数（序列削减的总步数）
     *   - simulation_idx: 当前模拟索引（序列削减的当前阶段，从0开始）
     *   - gumble_noise: 该样本的Gumbel噪声向量（长度=动作数）
     *   - current_num_top_actions: 当前保留的顶部候选动作数量
     *
     * 返回：选中的动作索引
     * =====================================================================
     */
    int select_action(CNode* node, tools::CMinMaxStats &min_max_stats,
                      int num_simulations, int simulation_idx,
                      const std::vector<float>& gumble_noise,
                      int current_num_top_actions){

        int action = -1;
        int num_actions = node->legal_actions.size();

        // Check if this is a root node by examining if selected_children_idx is populated
        // (Root nodes go through Sequential Ha2lving and have selected_children_idx set)
        bool is_root = !node->selected_children_idx.empty() || node->visit_count == 0;

        if(is_root){
            // ============================================================
            // 【根节点处理】Sequential Halving + 等量访问策略
            // ============================================================

            if(simulation_idx == 0){
                // 【第一阶段 (simulation_idx == 0)】
                // 目的：基于Gumbel-Max采样初始化顶部候选动作集合

                // 1. 获取根节点所有子节点的先验概率（来自网络）
                std::vector<float> children_prior = node->get_children_priors();

                // 2. 计算初始得分 = Gumbel噪声 + log(先验)
                //    这实现了Gumbel-Max采样：max_a[g_a + log(p_a)]
                //    Gumbel噪声用于探索，先验用于指导
                std::vector<float> children_scores;
                for(int a = 0; a < num_actions; ++a){
                    // g_a: Gumbel噪声(从高斯分布生成)
                    // p_a: 网络输出的先验策略
                    children_scores.push_back(gumble_noise[a] + children_prior[a]);
                }

                // 3. 对得分进行降序排序，获取排序后的动作索引
                std::vector<size_t> idx(children_scores.size());
                std::iota(idx.begin(), idx.end(), 0);  // idx初始化为[0, 1, 2, ...]
                std::sort(idx.begin(), idx.end(),
                         [&children_scores](size_t i1, size_t i2) {
                             return children_scores[i1] > children_scores[i2];
                         });

                // 4. 保存前K个最佳动作到节点中（作为候选集合）
                //    【关键】这些候选在后续迭代中会被等量访问
                //    实现了序列削减的"筛选"阶段
                node->selected_children_idx.clear();
                for(int a = 0; a < current_num_top_actions; ++a){
                    node->selected_children_idx.push_back(idx[a]);
                }
            }

            // 【后续阶段 (simulation_idx > 0)】
            // 使用等量访问策略：从候选集合中轮流选择动作
            // 目的：平衡评估各个候选动作，同时让高价值候选更早收敛
            action = node->do_equal_visit(num_simulations);
        }
        else{
            // ============================================================
            // 【非根节点处理】改进策略 + 访问计数平衡
            // ============================================================

            // 1. 计算所有子节点的【变换后完整Q值】
            //    完整Q = R + γ*V (当前节点的累积奖励 + 折扣未来值)
            //    通过min-max统计进行归一化到[0,1]范围内
            std::vector<float> transformed_completed_Qs =
                get_transformed_completed_Qs(node, min_max_stats, 0);

            // 2. 计算【改进策略】= softmax(transformed_Q)
            //    这是MCTS搜索中学到的改进策略，优于网络初始策略
            std::vector<float> improved_policy =
                node->get_improved_policy(transformed_completed_Qs);

            // 3. 获取原始网络先验策略（用于对比或调试）
            std::vector<float> ori_policy = node->get_children_priors();

            // 4. 获取每个子节点的访问次数（反映已探索程度）
            std::vector<int> children_visits = node->get_children_visits();

            // 5. 计算每个动作的【选择得分】
            //    得分 =    
            //    其中：访问惩罚 = 子节点访问次数 / (1 + 父节点访问次数)
            //
            //    设计原理：
            //    - 改进策略项：偏向高Q值的动作（利用）
            //    - 访问惩罚项：偏向访问较少的动作（探索）
            //    - 除以父节点访问次数：动态平衡，搜索越深惩罚越大
            std::vector<float> children_scores(num_actions, 0.0);
            for(int a = 0; a < num_actions; ++a){
                float visit_penalty = children_visits[a] / (1.0f + float(node->visit_count));
                float score = improved_policy[a] - visit_penalty;
                children_scores[a] = score;
            }

            // 6. 选择得分最高的动作（贪心选择）
            action = argmax(children_scores);
        }

        return action;
    }

    // ========== MuZero/UCB 风格的批量遍历（备份版本） ==========
    void cbatch_traverse_ucb(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch)
    {
        /*
        Overview:
            Search node path from the roots using UCB/PUCT selection (MuZero style).
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
        {
            players = 1;
        }
        else
        {
            players = 2;
        }
        
        for (int i = 0; i < results.num; ++i)
        {
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);
            
            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                is_root = 0;
                parent_q = mean_q;
                
                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players);
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                    {
                        virtual_to_play_batch[i] = 2;
                    }
                    else
                    {
                        virtual_to_play_batch[i] = 1;
                    }
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

    /**
     * =====================================================================
     * 【EfficientZero V2 核心】批量树遍历函数 - cbatch_traverse
     * =====================================================================
     * 该函数是 EfficientZero V2 MCTS 搜索的核心遍历算法，集成了 Sequential Halving：
     *   1. 对批次中的每个样本进行树遍历
     *   2. 使用 select_action() 进行动作选择（集成 Sequential Halving）
     *   3. 遍历直至到达叶节点（未扩展的节点）
     *   4. 记录搜索路径、动作序列、叶节点等信息供后续回传使用
     *
     * 参数说明：
     *   roots                  - 批量样本的根节点集合（CRoots对象数组）
     *   min_max_stats_lst      - 批量min-max统计对象集合（用于Q值归一化）
     *   results                - 搜索结果缓冲区（保存搜索路径、动作、叶节点等）
     *   num_simulations        - 总模拟次数（序列削减的总步数）
     *   simulation_idx         - 当前模拟索引（当前处于第几次迭代）
     *   gumble_noise          - 批量Gumbel噪声矩阵（大小：batch_size × num_actions）
     *   current_num_top_actions - 当前保留的顶部候选动作数（随削减递减）
     *   virtual_to_play_batch  - 虚拟玩家批次（用于两人游戏，单人游戏时为[-1,-1,...]）
     * =====================================================================
     */
    void cbatch_traverse(CRoots *roots,
                         tools::CMinMaxStatsList *min_max_stats_lst,
                         CSearchResults &results,
                         int num_simulations,
                         int simulation_idx,
                         const std::vector<std::vector<float>>& gumble_noise,
                         int current_num_top_actions,
                         std::vector<int> &virtual_to_play_batch) {

        // 初始化结果容器
        int last_action = -1;
        results.search_lens = std::vector<int>();

        // 判断游戏类型（单人/双人）
        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end());
        if (largest_element == -1) {
            players = 1;  // 单人游戏
        } else {
            players = 2;  // 双人游戏
        }

        // 对批次中每个样本独立遍历
        for (int i = 0; i < results.num; ++i) {
            CNode *node = &(roots->roots[i]);
            int search_len = 0;
            results.search_paths[i].push_back(node);

            // 从根节点遍历到叶节点
            while (node->expanded()) {
                // 【关键】使用 select_action 替代 cselect_child
                // 该函数内部集成了 Sequential Halving 逻辑：
                //   - 根节点：在 simulation_idx==0 时初始化候选集，后续等量访问
                //   - 非根节点：使用改进策略 + 访问计数平衡
                int action = select_action(
                    node,
                    min_max_stats_lst->stats_lst[i],
                    num_simulations,           // Sequential Halving 参数
                    simulation_idx,            // Sequential Halving 参数
                    gumble_noise[i],          // Sequential Halving 参数
                    current_num_top_actions   // Sequential Halving 参数
                );

                // 【两人游戏兼容】切换玩家
                if (players > 1) {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1) {
                        virtual_to_play_batch[i] = 2;
                    } else {
                        virtual_to_play_batch[i] = 1;
                    }
                }

                node->best_action = action;
                node = node->get_child(action);
                last_action = action;
                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            // 记录叶节点父节点的信息
            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            // 记录父节点的隐状态索引位置（后续用于神经网络推理定位隐状态）
            results.latent_state_index_in_search_path.push_back(parent->current_latent_state_index);
            results.latent_state_index_in_batch.push_back(parent->batch_index);

            // 记录此次搜索的其他信息
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
            Search node path from the roots.
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
        {
            players = 1;
        }
        else
        {
            players = 2;
        }

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
                    {
                        virtual_to_play_batch[i] = 2;
                    }
                    else
                    {
                        virtual_to_play_batch[i] = 1;
                    }
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

    // ==================== EfficientZero V2 Sequential Halving 实现 ====================

    // Softmax 函数：将 logits 转换为概率分布
    std::vector<float> softmax(const std::vector<float> &logits){
        std::vector<float> policy(logits.size(), 0.0);

        // 数值稳定：减去最大值防止溢出
        float max_logit = -1e9;
        for(size_t a = 0; a < logits.size(); ++a){
            if(logits[a] > max_logit) max_logit = logits[a];
        }

        for(size_t a = 0; a < logits.size(); ++a){
            policy[a] = exp(logits[a] - max_logit);
        }

        // 归一化
        float policy_sum = 0.0;
        for(size_t a = 0; a < policy.size(); ++a){
            policy_sum += policy[a];
        }
        for(size_t a = 0; a < policy.size(); ++a){
            policy[a] = policy[a] / policy_sum;
        }
        return policy;
    }

    // 获取子节点的先验概率
    std::vector<float> CNode::get_children_priors(){
        std::vector<float> priors(this->legal_actions.size(), 0.0);
        for(size_t i = 0; i < this->legal_actions.size(); ++i){
            int action = this->legal_actions[i];
            priors[i] = this->children[action].prior;
        }
        return priors;
    }

    /**
     * =====================================================================
     * 获取子节点访问次数向量
     * =====================================================================
     * 将该节点的所有子节点的访问次数合并为一个向量
     * 向量大小与合法动作数量相同
     *
     * 返回：visit_count 向量，索引对应合法动作顺序
     * =====================================================================
     */
    std::vector<int> CNode::get_children_visits(){
        // 创建与合法动作数量相同大小的向量，初始化为0
        std::vector<int> visits(this->legal_actions.size(), 0);

        // 遍历所有合法动作
        for(size_t i = 0; i < this->legal_actions.size(); ++i){
            // 获取该动作的索引
            int action = this->legal_actions[i];

            // 从子节点中获取访问次数
            visits[i] = this->children[action].visit_count;
        }

        return visits;
    }

    // 获取该节点的奖励
    float CNode::get_reward(){
        if(this->is_reset){
            // 如果是重置点，直接返回 value_prefix
            return this->value_prefix;
        } else {
            // 否则返回与父节点的差值
            return this->value_prefix - this->parent_value_prefix;
        }
    }

    // 获取单个动作的 Q 值：Q(s,a) = R(a) + γ*V(s')
    float CNode::get_qsa(int action){
        CNode* child = &(this->children[action]);
        // Q(s,a) = R(a) + γ*V(s')
        float qsa = child->get_reward() + this->discount * child->value();
        return qsa;
    }

    // 计算混合值估计（用于未访问子节点的乐观估计）
    float CNode::get_v_mix(){
        // 1. 获取当前节点的策略分布
        std::vector<float> priors = this->get_children_priors();
        std::vector<float> pi_lst = softmax(priors);

        float pi_sum = 0.0;        // 已访问动作的策略概率和
        float pi_qsa_sum = 0.0;    // π(a) * Q(a) 的加权和

        // 2. 遍历所有合法动作，只统计已扩展的子节点
        for(size_t i = 0; i < this->legal_actions.size(); ++i){
            int action = this->legal_actions[i];
            if(this->children[action].expanded()){
                // 子节点已访问：累加策略概率
                pi_sum += pi_lst[i];
                // 累加加权 Q 值
                pi_qsa_sum += pi_lst[i] * this->get_qsa(action);
            }
        }

        // 3. 计算 v_mix
        float v_mix = 0.0;
        const float EPSILON = 0.000001;

        if(pi_sum < EPSILON) {
            // 没有子节点被访问，直接用节点值
            v_mix = this->value();
        }
        else{
            // v_mix = (1/(1+N)) * [V + N * Σ(π*Q) / Σ(π)]
            v_mix = (1.0 / (1.0 + this->visit_count)) * (this->value() + this->visit_count * pi_qsa_sum / pi_sum);
        }

        return v_mix;
    }

    // 获取完整的Q值（包括已访问和未访问节点的估计）
    std::vector<float> CNode::get_completed_Q(tools::CMinMaxStats &min_max_stats, int to_normalize){
        // 1. 创建返回值向量
        std::vector<float> completed_Qs(this->legal_actions.size(), 0.0);

        // 2. 计算混合值（用于未访问的子节点）
        float v_mix = this->get_v_mix();

        // 3. 对每个动作计算完整的 Q 值
        for(size_t i = 0; i < this->legal_actions.size(); ++i){
            int action = this->legal_actions[i];
            float Q = 0.0;

            // 判断子节点是否已扩展（访问过）
            if(this->children[action].expanded()){
                // 子节点已访问：Q = R + γ*V
                Q = this->get_qsa(action);
            }
            else {
                // 子节点未访问：用 v_mix 乐观估计
                Q = v_mix;
            }

            // 4. 根据 to_normalize 标志选择归一化方式
            if (to_normalize == 1) {
                // 模式1：使用 min-max 统计进行归一化
                completed_Qs[i] = min_max_stats.normalize(Q);
                if (completed_Qs[i] < 0.0) completed_Qs[i] = 0.0;
                if (completed_Qs[i] > 1.0) completed_Qs[i] = 1.0;
            }
            else {
                completed_Qs[i] = Q;
            }
        }

        // 5. 如果选择最终归一化模式（to_normalize == 2）
        if (to_normalize == 2){
            float v_max = -1e9;
            float v_min = 1e9;
            for(float q : completed_Qs){
                if(q > v_max) v_max = q;
                if(q < v_min) v_min = q;
            }

            if(v_max > v_min){
                for(size_t i = 0; i < completed_Qs.size(); ++i){
                    completed_Qs[i] = (completed_Qs[i] - v_min) / (v_max - v_min);
                }
            }
        }

        return completed_Qs;
    }

    // 获取改进后的策略（基于 Q 值）
    std::vector<float> CNode::get_improved_policy(const std::vector<float> &transformed_completed_Qs){
        // 获取子节点的先验
        std::vector<float> logits = this->get_children_priors();

        // EfficientZero V2 改进：logits = prior + transformed_Q
        for(size_t i = 0; i < logits.size(); ++i){
            logits[i] = logits[i] + transformed_completed_Qs[i];
        }

        // 转换为概率分布
        std::vector<float> policy = softmax(logits);
        return policy;
    }

    /**
     * =====================================================================
     * 【EfficientZero V2 根节点策略】等量访问函数 - do_equal_visit
     * =====================================================================
     * 在 Sequential Halving 中，根节点采用等量访问策略来评估候选动作。
     * 该函数从已筛选的候选集合中选择访问次数最少的动作。
     *
     * 算法原理：
     *   1. 遍历 selected_children_idx 中所有候选动作
     *   2. 找到访问次数最少的动作
     *   3. 返回该动作的索引
     *
     * 目的：确保在同一阶段中各个候选动作被公平地评估
     *
     * 参数：
     *   num_simulations - 总模拟数（用于初始化最小值）
     *
     * 返回：访问次数最少的候选动作索引
     * =====================================================================
     */
    int CNode::do_equal_visit(int num_simulations){
        // 初始化最小访问次数为一个很大的值（num_simulations + 1）
        // 这样确保任何实际的访问次数都会更小
        int min_visit_count = num_simulations + 1;
        int action = -1;  // 初始化为-1，表示未选择

        // 遍历所有候选动作（在 selected_children_idx 中）
        for(int selected_child_idx : this->selected_children_idx){
            // 获取该候选动作的访问次数
            int visit_count = (this->get_child(selected_child_idx))->visit_count;

            // 如果该动作的访问次数更少，更新最小访问次数和选中的动作
            if(visit_count < min_visit_count){
                action = selected_child_idx;
                min_visit_count = visit_count;
            }
        }

        // 返回选中的动作（访问次数最少的候选）
        return action;
    }

    // ==================== 辅助函数 ====================

    // 获取向量中的最大浮点数
    float max_float(const std::vector<float> &arr){
        if(arr.empty()) return -1e9;
        float max_val = arr[0];
        for(size_t i = 1; i < arr.size(); ++i){
            if(arr[i] > max_val) max_val = arr[i];
        }
        return max_val;
    }

    // 获取向量中的最小浮点数
    float min_float(const std::vector<float> &arr){
        if(arr.empty()) return 1e9;
        float min_val = arr[0];
        for(size_t i = 1; i < arr.size(); ++i){
            if(arr[i] < min_val) min_val = arr[i];
        }
        return min_val;
    }

    // 获取向量中的最大整数
    int max_int(const std::vector<int> &arr){
        if(arr.empty()) return -1e9;
        int max_val = arr[0];
        for(size_t i = 1; i < arr.size(); ++i){
            if(arr[i] > max_val) max_val = arr[i];
        }
        return max_val;
    }

    // 求浮点数向量的和
    float sum_float(const std::vector<float> &arr){
        float res = 0.0;
        for(float a : arr) res += a;
        return res;
    }

    // 求整数向量的和
    int sum_int(const std::vector<int> &arr){
        int res = 0;
        for(int a : arr) res += a;
        return res;
    }

    // 获取最大值的索引
    int argmax(const std::vector<float> &arr){
        if(arr.empty()) return -1;
        int index = 0;
        float max_val = arr[0];
        for(size_t i = 1; i < arr.size(); ++i){
            if(arr[i] > max_val){
                max_val = arr[i];
                index = i;
            }
        }
        return index;
    }

    // 获取转换后的完整 Q 值（带 Sigma 变换）
    std::vector<float> get_transformed_completed_Qs(CNode* node, tools::CMinMaxStats &min_max_stats, int final){
        // 1. 获取完整Q值（根据 final 参数选择归一化模式）
        int to_normalize = (final == 0) ? 1 : 2;
        std::vector<float> completed_Qs = node->get_completed_Q(min_max_stats, to_normalize);

        // 2. 计算最大访问数
        int max_child_visit_count = 0;
        for(int action : node->legal_actions){
            if(node->children.count(action) > 0){
                int visit_count = node->children[action].visit_count;
                if(visit_count > max_child_visit_count){
                    max_child_visit_count = visit_count;
                }
            }
        }

        // 3. Sigma 变换（缩放）：Q' = (c_visit + max_visit) * c_scale * Q
        for(size_t i = 0; i < completed_Qs.size(); ++i){
            completed_Qs[i] = (min_max_stats.c_visit + max_child_visit_count) * min_max_stats.c_scale * completed_Qs[i];
        }

        return completed_Qs;
    }

    // Sequential Halving：逐步淘汰差的动作
    int sequential_halving(CNode* root, const std::vector<float>& gumbel_noise,
                          tools::CMinMaxStats &min_max_stats, int current_phase, int current_num_top_actions){
        // 1. 获取子节点的先验概率和转换后的Q值
        std::vector<float> children_prior = root->get_children_priors();
        std::vector<float> transformed_completed_Qs = get_transformed_completed_Qs(root, min_max_stats, 0);

        // 获取当前已选的动作列表
        std::vector<int> selected_children_idx = root->selected_children_idx;
        std::vector<float> children_scores;

        // 2. 计算分数：gumbel噪声 + 先验 + 转换后的Q值
        // 直接用 action 作为索引（action ∈ [0, num_actions-1]）
        for(int action : selected_children_idx){
            float score = gumbel_noise[action] + children_prior[action] + transformed_completed_Qs[action];
            children_scores.push_back(score);
        }

        // 3. 创建索引数组并初始化为 [0, 1, 2, ...]
        std::vector<size_t> idx(children_scores.size());
        std::iota(idx.begin(), idx.end(), 0);

        // 4. 按分数从高到低排序索引
        std::sort(idx.begin(), idx.end(),
                  [&children_scores](size_t index_1, size_t index_2) {
                      return children_scores[index_1] > children_scores[index_2];
                  });

        // 5. 清空已选动作，只保留分数最高的 top-m 个
        root->selected_children_idx.clear();
        int keep_count = std::min(current_num_top_actions, (int)selected_children_idx.size());
        for(int i = 0; i < keep_count; ++i){
            root->selected_children_idx.push_back(selected_children_idx[idx[i]]);
        }

        // 6. 返回分数最高的动作
        int best_action = root->selected_children_idx[0];
        return best_action;
    }

    // 批量 Sequential Halving：对多个搜索树进行动作选择
    std::vector<int> c_batch_sequential_halving(CRoots *roots, const std::vector<std::vector<float>>& gumbel_noises,
                                               tools::CMinMaxStatsList *min_max_stats_lst,
                                               int current_phase, int current_num_top_actions){
        std::vector<int> best_actions(roots->root_num, -1);

        for(int i = 0; i < roots->root_num; ++i){
            int action = sequential_halving(&(roots->roots[i]), gumbel_noises[i],
                                           min_max_stats_lst->stats_lst[i], current_phase, current_num_top_actions);
            best_actions[i] = action;
        }

        return best_actions;
    }
}