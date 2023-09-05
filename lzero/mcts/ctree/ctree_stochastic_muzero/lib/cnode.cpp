// C++11

#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <cassert>
#include <numeric>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <iterator>

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
        this->is_chance = false;
        this->chance_space_size= 2;

    }

    CNode::CNode(float prior, std::vector<int> &legal_actions, bool is_chance, int chance_space_size)
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
        this->is_chance = is_chance;
        this->chance_space_size = chance_space_size;
    }

    CNode::~CNode() {}

    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits, bool child_is_chance)
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


        // assert((this->is_chance != child_is_chance) && "is_chance and child_is_chance should be different");
        
        if(this->is_chance == true){
            child_is_chance = false;
            this->reward = 0.0;
        }
        else{
            child_is_chance = true;
        }

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
            this->children[a] = CNode(prior, tmp_empty, child_is_chance, this->chance_space_size); // only for muzero/efficient zero, not support alphazero
            // this->children[a] = CNode(prior, tmp_empty, is_chance = child_is_chance); // only for muzero/efficient zero, not support alphazero
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
        Outputs:
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
        Outputs:
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

    CRoots::CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list, int chance_space_size=2)
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
            this->roots.push_back(CNode(0, this->legal_actions_list[i], false, chance_space_size));
            // this->roots.push_back(CNode(0, this->legal_actions_list[i], false));

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
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i], true);
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
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i], true);

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
        Outputs:
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
        Outputs:
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
        while (node_stack.size() > 0)
        {
            CNode *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                //                # NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
                //                # but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                //                # true_reward = node.value_prefix - (- parent_value_prefix)
                //                float true_reward = node->value_prefix - node->parent_value_prefix;
                float true_reward = node->reward;

                float qsa;
                if (players == 1)
                    qsa = true_reward + discount_factor * node->value();
                else if (players == 2)
                    // TODO(pu):
                    qsa = true_reward + discount_factor * (-1) * node->value();

                min_max_stats.update(qsa);
            }

            for (auto a : node->legal_actions)
            {
                CNode *child = node->get_child(a);
                if (child->expanded())
                {
                    node_stack.push(child);
                }
            }
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
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float true_reward = node->reward;

                min_max_stats.update(true_reward + discount_factor * node->value());

                bootstrap_value = true_reward + discount_factor * bootstrap_value;
                // std::cout << "to_play: " << to_play << std::endl;

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

    void cbatch_backpropagate(int current_latent_state_index, float discount_factor, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch, std::vector<bool> &is_chance_list, std::vector<int> &leaf_idx_list)
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
        */

        if (leaf_idx_list.empty()) {
            leaf_idx_list.resize(results.num);
            for (int i = 0; i < results.num; ++i) {
                leaf_idx_list[i] = i;
            }
        }

        for (auto leaf_order = 0; leaf_order < leaf_idx_list.size(); ++leaf_order) {
            int i = leaf_idx_list[leaf_order];
        }
        for (int leaf_order = 0; leaf_order < leaf_idx_list.size(); ++leaf_order)
        {
            int i = leaf_idx_list[leaf_order];
            results.nodes[i]->expand(to_play_batch[i], current_latent_state_index, i, value_prefixs[leaf_order], policies[leaf_order], is_chance_list[i]);
            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[leaf_order], discount_factor);
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
        Outputs:
            - action: the action to select.
        */
        if (root->is_chance) {
                // std::cout << "root->is_chance: True " << std::endl;

                // If the node is a chance node, we sample from the prior outcome distribution.
                std::vector<int> outcomes;
                std::vector<double> probs;

                for (const auto& kv : root->children) {
                    outcomes.push_back(kv.first);
                    probs.push_back(kv.second.prior); // Assuming 'prior' is a member variable of Node
                }

                std::random_device rd;
                std::mt19937 gen(rd());
                std::discrete_distribution<> dist(probs.begin(), probs.end());

                int outcome = outcomes[dist(gen)];
                // std::cout << "Outcome: " << outcome << std::endl;

            return outcome;
        }

        // std::cout << "root->is_chance: False " << std::endl;

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
        Outputs:
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

            // std::cout << "root->is_chance: " <<node->is_chance<< std::endl;
            // node->is_chance=false;

            while (node->expanded())
            {
                float mean_q = node->compute_mean_q(is_root, parent_q, discount_factor);
                is_root = 0;
                parent_q = mean_q;
                // std::cout << "node->is_chance: " <<node->is_chance<< std::endl;

                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players);
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
            results.leaf_node_is_chance.push_back(node->is_chance);
            results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);

        }
    }

}