#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <iostream>
#include <math.h>

template <class T>
size_t hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return seed;
}

namespace tree{
    //*********************************************************

    Action::Action(){
        this->is_root_action = 0;
    }

    Action::Action(std::vector<float>, int is_root_action){
        this->value = value;
        this->is_root_action = is_root_action;
    }

    Action::~Action(){}

    std::vector<size_t> Action::get_hash( void ){
        std::vector<size_t> hash;
        for(int i =0; i<this->value.size(); ++i){
            std::size_t hash_i = std::hash<std::string>() (std::to_string(this->value[i]));
            hash.push_back(hash_i);
        }
        return hash;
    }
    size_t Action::get_combined_hash( void ){
        std::vector<size_t> hash=this->get_hash();
        size_t combined_hash=hash[0];
        for(int i =1; i<hash.size(); ++i){
            combined_hash = hash_combine(combined_hash,hash[i]);
        }
        return combined_hash;
    }

    //*********************************************************

    CSearchResults::CSearchResults(){
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num){
        this->num = num;
        for(int i = 0; i < num; ++i){
            this->search_paths.push_back(std::vector<CNode*>());
        }
    }

    CSearchResults::~CSearchResults(){}

    //*********************************************************

    CNode::CNode(){
        this->prior = 0;
//        this->legal_actions = Action(-1);
//        this->legal_actions = -1;
        this->action_space_size = 9;
        this->num_of_sampled_actions = 20;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
//        this->best_action = -1;
//        Action best_action = Action(-1);
        Action best_action;
        this->best_action = best_action;

        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
    }

    CNode::CNode(float prior, std::vector<Action> &legal_actions, int  action_space_size, int num_of_sampled_actions){
        this->prior = prior;
        this->legal_actions = legal_actions;
        this->action_space_size = action_space_size;
        this->num_of_sampled_actions = num_of_sampled_actions;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
//        this->best_action = -1;
//        this->best_action = Action(-1);
        this->to_play = 0;
        this->value_prefix = 0.0;
        this->parent_value_prefix = 0.0;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
    }

    CNode::~CNode(){}

    void CNode::expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits){
        this->to_play = to_play;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->value_prefix = value_prefix;

        int action_num = policy_logits.size();
        std::vector<int> all_actions;
         for(int i =0; i<action_num; ++i){
            all_actions.push_back(i);
        }
//        if(this->legal_actions.size()==0)
//            for (int i = 0; i < 1; ++i){
//              }
//            this->legal_actions.assign(all_actions.begin(), all_actions.end());

        this->action_space_size = policy_logits.size()/2;
//        std::vector<float> mu = policy_logits[: self.action_space_size];
        std::vector<float> mu;
        std::vector<float> sigma;
        for(int i = 0; i < this->action_space_size; ++i){
            mu.push_back(policy_logits[i]);
            sigma.push_back(policy_logits[this->action_space_size + i]);
        }

        // 从epoch（1970年1月1日00:00:00 UTC）开始经过的纳秒数，unsigned类型会截断这个值
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        std::vector<std::vector<float> > sampled_actions;
        float sampled_action_tmp;
        std::vector<float> sampled_action;
        std::vector<float> sampled_actions_log_prob;
        std::default_random_engine generator(seed);
      // 第一个参数为高斯分布的平均值，第二个参数为标准差
      //      std::normal_distribution<double> distribution(mu, sigma);
        for (int i = 0; i < this->num_of_sampled_actions; ++i){
            float sampled_action_log_prob = 1;
                for (int j = 0; j < this->action_space_size; ++j){
                    std::normal_distribution<double> distribution(mu[j], sigma[j]);
                    sampled_action_tmp = distribution(generator);
                    // refer to python normal log_prob method
                    sampled_action_log_prob *= - pow((sampled_action_tmp - mu[j]), 2) / (2 * pow(sigma[j], 2)) -  log(mu[j]) - log(sqrt(2 * M_PI));
                    sampled_action.push_back(sampled_action_tmp);
                }
                sampled_actions.push_back(sampled_action);
                sampled_actions_log_prob.push_back(sampled_action_log_prob);
        }

//        std::map<int, float> policy;
//        for(int i =0; i<this->num_of_sampled_actions; ++i){
//            this->policy[i] = sampled_actions_log_prob[i];
//        }

        float prior;
        for (int i = 0; i < this->num_of_sampled_actions; ++i){
//            prior = policy[a] / policy_sum;
            Action action = Action(sampled_actions[i], 0);
            std::vector<Action> legal_actions;
//            for (int i = 0; i < 1; ++i){
//
//              }
            this->children[action] = CNode(sampled_actions_log_prob[i], legal_actions, this->action_space_size, this->num_of_sampled_actions); // only for muzero/efficient zero, not support alphazero
            this->legal_actions.push_back(action);
        }




//        float temp_policy;
//        float policy_sum = 0.0;
//        float policy[action_num];
//        float policy_max = FLOAT_MIN;
//        for(auto a: this->legal_actions){
//            if(policy_max < policy_logits[a]){
//                policy_max = policy_logits[a];
//            }
//        }
//
//        for(auto a: this->legal_actions){
//            temp_policy = exp(policy_logits[a] - policy_max);
//            policy_sum += temp_policy;
//            policy[a] = temp_policy;
//        }
//
//        float prior;
//        for(auto a: this->legal_actions){
//            prior = policy[a] / policy_sum;
//            this->children[a] = CNode(prior, all_actions); // only for muzero/efficient zero, not support alphazero
//        }
    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises){
        float noise, prior;
//        for(auto a: this->legal_actions){
//            noise = noises[a];
//            CNode* child = this->get_child(a);
        for(int i =0; i<this->legal_actions.size(); ++i){
            noise = noises[i];
            CNode* child = this->get_child(this->legal_actions[i]);

            prior = child->prior;
            child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
        }
    }

    float CNode::get_mean_q(int isRoot, float parent_q, float discount){
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        float parent_value_prefix = this->value_prefix;
        for(auto a: this->legal_actions){
            CNode* child = this->get_child(a);
            if(child->visit_count > 0){
                float true_reward = child->value_prefix - parent_value_prefix;
                if(this->is_reset == 1){
                    true_reward = child->value_prefix;
                }
                float qsa = true_reward + discount * child->value();
                total_unsigned_q += qsa;
                total_visits += 1;
            }
        }

        float mean_q = 0.0;
        if(isRoot && total_visits > 0){
            mean_q = (total_unsigned_q) / (total_visits);
        }
        else{
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
        }
        return mean_q;
    }

    void CNode::print_out(){
        return;
    }

    int CNode::expanded(){
        return this->children.size()>0;
    }

    float CNode::value(){
        float true_value = 0.0;
        if(this->visit_count == 0){
            return true_value;
        }
        else{
            true_value = this->value_sum / this->visit_count;
            return true_value;
        }
    }

    std::vector<Action> CNode::get_trajectory(){
        std::vector<Action> traj;

        CNode* node = this;
        Action best_action = node->best_action;
        while(best_action.is_root_action != 1){
            traj.push_back(best_action);
//            best_action = Action(best_action);
            node = node->get_child(best_action);
            best_action = node->best_action;
        }
        return traj;
    }

    std::vector<int> CNode::get_children_distribution(){
        std::vector<int> distribution;
        if(this->expanded()){
            for(auto a: this->legal_actions){
                CNode* child = this->get_child(a);
                distribution.push_back(child->visit_count);
            }
        }
        return distribution;
    }

    CNode* CNode::get_child(Action action){
        return &(this->children[action]);
    }

    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
//        this->pool_size = 0;
        this->num_of_sampled_actions = 20;
    }

//    CRoots::CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list, int action_space_size, int num_of_sampled_actions){
    CRoots::CRoots(int root_num, std::vector<std::vector<Action> > &legal_actions_list, int action_space_size, int num_of_sampled_actions){

        this->root_num = root_num;
//        this->pool_size = pool_size;
        this->legal_actions_list = legal_actions_list;
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->action_space_size = action_space_size;

        for(int i = 0; i < root_num; ++i){
//            this->roots.push_back(CNode(0, this->legal_actions_list[i], this->num_of_sampled_actions));
            std::vector<Action> legal_actions;
            this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float> > &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(to_play_batch[i], 0, i, value_prefixs[i], policies[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->roots.clear();
    }

//    std::vector<std::vector<int> > CRoots::get_trajectories(){
//        std::vector<std::vector<int> > trajs;
    std::vector<std::vector<Action> >* CRoots::get_trajectories(){

        std::vector<std::vector<Action> > trajs;
        trajs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return &trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions(){
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<float> CRoots::get_values(){
        std::vector<float> values;
        for(int i = 0; i < this->root_num; ++i){
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    //*********************************************************
    //
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount, int players){
        std::stack<CNode*> node_stack;
        node_stack.push(root);
        float parent_value_prefix = 0.0;
        int is_reset = 0;
        while(node_stack.size() > 0){
            CNode* node = node_stack.top();
            node_stack.pop();

            if(node != root){
//                # NOTE: in 2 player mode, value_prefix is not calculated according to the perspective of current player of node,
//                # but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
//                # true_reward = node.value_prefix - (- parent_value_prefix)
                float true_reward = node->value_prefix - node->parent_value_prefix;

                if(is_reset == 1){
                    true_reward = node->value_prefix;
                }
                float qsa;
                if(players == 1)
                    qsa = true_reward + discount * node->value();
                else if(players == 2)
                    // TODO(pu):
                     qsa = true_reward + discount * (-1) * node->value();

                min_max_stats.update(qsa);
            }

            for(auto a: node->legal_actions){
                CNode* child = node->get_child(a);
                if(child->expanded()){
                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }

            is_reset = node->is_reset;
        }
    }

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount){
        if(to_play == 0){
            float bootstrap_value = value;
            int path_len = search_path.size();
            for(int i = path_len - 1; i >= 0; --i){
                CNode* node = search_path[i];
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if(i >= 1){
                    CNode* parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
    //                float qsa = (node->value_prefix - parent_value_prefix) + discount * node->value();
    //                min_max_stats.update(qsa);
                }

                float true_reward = node->value_prefix - parent_value_prefix;
                min_max_stats.update(true_reward + discount * node->value());

                if(is_reset == 1){
                    // parent is reset
                    true_reward = node->value_prefix;
                }

                bootstrap_value = true_reward + discount * bootstrap_value;
            }
        }
        else
        {
            float bootstrap_value = value;
            int path_len = search_path.size();
            for(int i = path_len - 1; i >= 0; --i){
                CNode* node = search_path[i];
                if(node->to_play == to_play)
                    node->value_sum += bootstrap_value;
                else
                    node->value_sum += - bootstrap_value;
                node->visit_count += 1;

                float parent_value_prefix = 0.0;
                int is_reset = 0;
                if(i >= 1){
                    CNode* parent = search_path[i - 1];
                    parent_value_prefix = parent->value_prefix;
                    is_reset = parent->is_reset;
                }

                // NOTE: in 2 player mode, value_prefix is not calculated according to the perspective of current player of node,
               // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                float true_reward = node->value_prefix - parent_value_prefix;

                min_max_stats.update(true_reward + discount * node->value());

                if(is_reset == 1){
                    // parent is reset
                    true_reward = node->value_prefix;
                }
                if(node->to_play == to_play)
                    bootstrap_value = - true_reward + discount * bootstrap_value;
                else
                    bootstrap_value = true_reward + discount * bootstrap_value;
//                if(node->to_play == to_play)
//                    bootstrap_value = true_reward + discount * bootstrap_value;
//                else
//                    bootstrap_value = - true_reward + discount * bootstrap_value;
            }
        }
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst, std::vector<int> &to_play_batch){
        for(int i = 0; i < results.num; ++i){
            results.nodes[i]->expand(to_play_batch[i], hidden_state_index_x, i, value_prefixs[i], policies[i]);
            // reset
            results.nodes[i]->is_reset = is_reset_lst[i];

            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount);
        }
    }

    Action cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players){
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<Action> max_index_lst;
        for(auto a: root->legal_actions){

            CNode* child = root->get_child(a);
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount, players);

            if(max_score < temp_score){
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if(temp_score >= max_score - epsilon){
                max_index_lst.push_back(a);
            }
        }

//        int action = 0;
        Action action;
        if(max_index_lst.size() > 0){
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
//        return &action;

    }

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount, int players){
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0){
            value_score = parent_mean_q;
        }
        else {
            float true_reward = child->value_prefix - parent_value_prefix;
            if(is_reset == 1){
                true_reward = child->value_prefix;
            }

            if(players == 1)
                value_score = true_reward + discount * child->value();
            else if(players == 2)
                value_score = true_reward + discount * (-child->value());
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0) value_score = 0;
        if (value_score > 1) value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        std::vector<float> null_value;
        for (int i = 0; i < 1; ++i){
            null_value.push_back(i+0.1);
        }
        Action last_action=Action(null_value,1);
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(),virtual_to_play_batch.end()); // 0 or 2
        if(largest_element==0)
            players = 1;
        else
            players = 2;

        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            results.search_paths[i].push_back(node);

            while(node->expanded()){
                float mean_q = node->get_mean_q(is_root, parent_q, discount);
                is_root = 0;
                parent_q = mean_q;

//                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players);
                Action action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players);
                if(players>1)
                {
                    if(virtual_to_play_batch[i] == 1)
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

            CNode* parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
        }
    }

}