#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <cmath>
#include <random> 
#include <numeric>

#ifdef _WIN32
#include "..\..\common_lib\utils.cpp"
#else
#include "../../common_lib/utils.cpp"
#endif

namespace tree{

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
        this->legal_actions = legal_actions;

        this->visit_count = 0;
        this->value_sum = 0;
        this->raw_value = 0;
        this->best_action = -1;
        this->to_play = 0;
//        this->value_prefix = 0.0;
        this->reward = 0.0;
//        this->parent_value_prefix = 0.0;
        this->gumbel_scale = 1.0;
        this->gumbel_rng=0.0;
    }

    CNode::CNode(float prior, std::vector<int> &legal_actions){
        this->prior = prior;
        this->legal_actions = legal_actions;

        this->visit_count = 0;
        this->value_sum = 0;
        this->raw_value = 0;
        this->best_action = -1;
        this->to_play = 0;
//        this->value_prefix = 0.0;
//        this->parent_value_prefix = 0.0;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
        this->gumbel_scale = 1.0;
        this->gumbel_rng=0.0;

        this->gumbel = generate_gumbel(this->gumbel_scale, this->gumbel_rng, legal_actions.size());
    }

    CNode::~CNode(){}

    /////////////////
    // gumbel muzero related code
    ////////////////

    void CNode::setdata(float reward, int visit_count, float value_sum, float raw_value, int action_num, std::vector<float> child_prior, \
        std::vector<float> child_reward, std::vector<int> child_visit_count, std::vector<float> child_value_sum, std::vector<float> child_raw_value){
        // only for test
        this->reward = reward;
        this->visit_count = visit_count;
        this->value_sum = value_sum;
        this->raw_value = raw_value;
        std::vector<int> legal_actions;
        for (unsigned i=0;i < action_num;i++){
            legal_actions.push_back(i);
        }
        this->legal_actions=legal_actions;
        this->gumbel = generate_gumbel(this->gumbel_scale, this->gumbel_rng, legal_actions.size());
        std::vector<int> void_int;
        std::vector<float> void_float;
        for(auto a: this->legal_actions){
            this->children[a] = CNode(child_prior[a], void_int);
            this->children[a].setdata(child_reward[a], child_visit_count[a], child_value_sum[a], child_raw_value[a], 0, \
                void_float, void_float, void_int, void_float, void_float);
        }
    }

    void CNode::expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward, float value, const std::vector<float> &policy_logits){
        this->to_play = to_play;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward = reward;
        this->raw_value = value;

        int action_num = policy_logits.size();
        if(this->legal_actions.size()==0){
            for(int i=0; i<action_num; ++i){
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
        for(auto a: this->legal_actions){
            if(policy_max < policy_logits[a]){
                policy_max = policy_logits[a];
            }
        }

        for(auto a: this->legal_actions){
            temp_policy = exp(policy_logits[a] - policy_max);
            policy_sum += temp_policy;
            policy[a] = temp_policy;
        }

        float prior;
        for(auto a: this->legal_actions){
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

    std::vector<float> CNode::get_q(float discount){
        std::vector<float> child_value;
        for(auto a: this->legal_actions){
            CNode* child = this->get_child(a);
            float true_reward = child->reward;
            float qsa = true_reward + discount * child->value();
            child_value.push_back(qsa);
        }
        return child_value;
    }

    float CNode::compute_mean_q(int isRoot, float parent_q, float discount){
        float total_unsigned_q = 0.0;
        int total_visits = 0;
//        float parent_value_prefix = this->value_prefix;
        for(auto a: this->legal_actions){
            CNode* child = this->get_child(a);
            if(child->visit_count > 0){
//                float true_reward = child->value_prefix - parent_value_prefix;
                float true_reward = child->reward;
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

    std::vector<int> CNode::get_trajectory(){
        std::vector<int> traj;

        CNode* node = this;
        int best_action = node->best_action;
        while(best_action >= 0){
            traj.push_back(best_action);

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

    CNode* CNode::get_child(int action){
        return &(this->children[action]);
    }

    std::vector<float> CNode::get_policy(float discount, int action_space_size){
        float infymin = -std::numeric_limits<float>::infinity();
        std::vector<int> child_visit_count;
        std::vector<float> child_prior;
        for(auto a: this->legal_actions){
            CNode* child = this->get_child(a);
            child_visit_count.push_back(child->visit_count);
            child_prior.push_back(child->prior);
        }
        assert(child_visit_count.size()==child_prior.size());
        std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(this, child_visit_count, child_prior, discount);
        std::vector<float> probs;
        for (int i=0;i<action_space_size;i++){
            probs.push_back(infymin);
        }
        for (int i=0;i<child_prior.size();i++){
            probs[this->legal_actions[i]] =  child_prior[i] + completed_qvalues[i];
        }
        csoftmax(probs, probs.size());

        return probs;
    }

    //*********************************************************

    CRoots::CRoots(){
        this->root_num = 0;
    }

    CRoots::CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list){
        this->root_num = root_num;
        this->legal_actions_list = legal_actions_list;

        for(int i = 0; i < root_num; ++i){
            this->roots.push_back(CNode(0, this->legal_actions_list[i]));
        }
    }

    CRoots::~CRoots(){}

    void CRoots::prepare(float root_noise_weight, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], values[i], policies[i]);
            this->roots[i].add_exploration_noise(root_noise_weight, noises[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &rewards, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch){
        for(int i = 0; i < this->root_num; ++i){
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], values[i], policies[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear(){
        this->roots.clear();
    }

    std::vector<std::vector<int> > CRoots::get_trajectories(){
        std::vector<std::vector<int> > trajs;
        trajs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions(){
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    std::vector<std::vector<float> > CRoots::get_policies(float discount, int action_space_size){
        std::vector<std::vector<float> > probs;
        probs.reserve(this->root_num);

        for(int i = 0; i < this->root_num; ++i){
            probs.push_back(this->roots[i].get_policy(discount, action_space_size));
        }
        return probs;
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
//        float parent_value_prefix = 0.0;
        while(node_stack.size() > 0){
            CNode* node = node_stack.top();
            node_stack.pop();

            if(node != root){
//                # NOTE: in 2 player mode, value_prefix is not calculated according to the perspective of current player of node,
//                # but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
//                # true_reward = node.value_prefix - (- parent_value_prefix)
//                float true_reward = node->value_prefix - node->parent_value_prefix;
                float true_reward = node->reward;

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
//                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }

        }
    }

    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount){
        assert(to_play == -1);
        float bootstrap_value = value;
        int path_len = search_path.size();
        for(int i = path_len - 1; i >= 0; --i){
            CNode* node = search_path[i];
            node->value_sum += bootstrap_value;
            node->visit_count += 1;

            float true_reward = node->reward;

            min_max_stats.update(true_reward + discount * node->value());

            bootstrap_value = true_reward + discount * bootstrap_value;
        }
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch){
        for(int i = 0; i < results.num; ++i){
            results.nodes[i]->expand(to_play_batch[i], hidden_state_index_x, i, value_prefixs[i], values[i], policies[i]);
            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount);
        }
    }

    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players){
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<int> max_index_lst;
        for(auto a: root->legal_actions){

            CNode* child = root->get_child(a);
            float temp_score = cucb_score(child, min_max_stats, mean_q, root->visit_count - 1,  pb_c_base, pb_c_init, discount, players);

            if(max_score < temp_score){
                max_score = temp_score;

                max_index_lst.clear();
                max_index_lst.push_back(a);
            }
            else if(temp_score >= max_score - epsilon){
                max_index_lst.push_back(a);
            }
        }

        int action = 0;
        if(max_index_lst.size() > 0){
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    /////////////////////////////
    // gumbel muzero related code
    /////////////////////////////

    int cselect_root_child(CNode* root, float discount, int num_simulations, int max_num_considered_actions){
        std::vector<int> child_visit_count;
        std::vector<float> child_prior;
        for(auto a: root->legal_actions){
            CNode* child = root->get_child(a);
            child_visit_count.push_back(child->visit_count);
            child_prior.push_back(child->prior);
        }
        assert(child_visit_count.size()==child_prior.size());

        std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(root, child_visit_count, child_prior, discount);
        std::vector<std::vector<int>> visit_table = get_table_of_considered_visits(max_num_considered_actions, num_simulations);
        
        int num_valid_actions = root->legal_actions.size();
        int num_considered = std::min(max_num_considered_actions, num_simulations);
        int simulation_index = std::accumulate(child_visit_count.begin(), child_visit_count.end(), 0);
        int considered_visit = visit_table[num_considered][simulation_index];

        std::vector<float> score = score_considered(considered_visit, root->gumbel, child_prior, completed_qvalues, child_visit_count);

        float argmax = -std::numeric_limits<float>::infinity();
        int max_action = root->legal_actions[0];
        int index = 0;
        for(auto a: root->legal_actions){
            if(score[index] > argmax){
                argmax = score[index];
                max_action = a;
            }
            index += 1;
        }

        return max_action;    
    }

    int cselect_interior_child(CNode* root, float discount){
        std::vector<int> child_visit_count;
        std::vector<float> child_prior;
        for(auto a: root->legal_actions){
            CNode* child = root->get_child(a);
            child_visit_count.push_back(child->visit_count);
            child_prior.push_back(child->prior);
        }
        assert(child_visit_count.size()==child_prior.size());
        std::vector<float> completed_qvalues = qtransform_completed_by_mix_value(root, child_visit_count, child_prior, discount);
        std::vector<float> probs;
        for (int i=0;i<child_prior.size();i++){
            probs.push_back(child_prior[i] + completed_qvalues[i]);
        }
        csoftmax(probs, probs.size());
        int visit_count_sum = std::accumulate(child_visit_count.begin(), child_visit_count.end(), 0);
        std::vector<float> to_argmax;
        for (int i=0;i<probs.size();i++){
            to_argmax.push_back(probs[i] - (float)child_visit_count[i]/(float)(1+visit_count_sum));
        }
        
        float argmax = -std::numeric_limits<float>::infinity();
        int max_action = root->legal_actions[0];
        int index = 0;
        for(auto a: root->legal_actions){
            if(to_argmax[index] > argmax){
                argmax = to_argmax[index];
                max_action = a;
            }
            index += 1;
        }
        
        return max_action;
    }
    //////////////////////////////

    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount, int players){
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        prior_score = pb_c * child->prior;
        if (child->visit_count == 0){
            value_score = parent_mean_q;
        }
        else {
            float true_reward = child->reward;
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

    void cbatch_traverse(CRoots *roots, int num_simulations, int max_num_considered_actions, float discount, CSearchResults &results, std::vector<int> &virtual_to_play_batch){
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        int last_action = -1;
        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(),virtual_to_play_batch.end()); // 0 or 2
        if(largest_element==-1)
            players = 1;
        else
            players = 2;

        for(int i = 0; i < results.num; ++i){
            CNode *node = &(roots->roots[i]);
            int is_root = 1;
            int search_len = 0;
            int action = 0;
            results.search_paths[i].push_back(node);

            while(node->expanded()){
                if(is_root){
                    action = cselect_root_child(node, discount, num_simulations, max_num_considered_actions);
                }
                else{
                    action = cselect_interior_child(node, discount);
                }
                is_root = 0;

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
            results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);

        }
    }

    /////////////////////////////
    // gumbel muzero related code
    /////////////////////////////

    void csoftmax(std::vector<float> &input, int input_len)
    {
        assert (input != NULL);
        assert (input_len != 0);
        int i;
        float m;
        /* Find maximum value from input array */
        m = input[0];
        for (i = 1; i < input_len; i++) {
            if (input[i] > m) {
                m = input[i];
            }
        }

        float sum = 0;
        for (i = 0; i < input_len; i++) {
            sum += expf(input[i]-m);
        }

        for (i = 0; i < input_len; i++) {
            input[i] = expf(input[i] - m - log(sum));
        }    
    }

    float compute_mixed_value(float raw_value, std::vector<float> q_values, std::vector<int> &child_visit, std::vector<float> &child_prior){
        float visit_count_sum = 0.0;
        float probs_sum = 0.0;
        float weighted_q_sum = 0.0;
        float min_num = -10e7;

        for(unsigned int i = 0;i < child_visit.size();i++)
            visit_count_sum += child_visit[i];

        for(unsigned int i = 0;i < child_prior.size();i++)
            // Ensuring non-nan prior
            child_prior[i] = std::max(child_prior[i], min_num);
        
        for(unsigned int i = 0;i < child_prior.size();i++)
            if (child_visit[i] > 0)
                probs_sum += child_prior[i];
        
        for (unsigned int i = 0;i < child_prior.size();i++)
            if (child_visit[i] > 0){
                weighted_q_sum += child_prior[i] * q_values[i] / probs_sum;
            }

        return (raw_value + visit_count_sum * weighted_q_sum) / (visit_count_sum+1);
    }

    void rescale_qvalues(std::vector<float> &value, float epsilon){
        float max_value = *max_element(value.begin(), value.end());
        float min_value = *min_element(value.begin(), value.end());
        float gap = max_value - min_value;
        gap = std::max(gap, epsilon);
        for (unsigned int i = 0;i < value.size();i++){
            value[i] = (value[i]-min_value)/gap;
        }
    }

    std::vector<float> qtransform_completed_by_mix_value(CNode *root, std::vector<int> & child_visit, \
        std::vector<float> & child_prior, float discount, float maxvisit_init, float value_scale, \
        bool rescale_values, float epsilon){
        assert (child_visit.size() == child_prior.size());
        std::vector<float> qvalues;
        std::vector<float> child_prior_tmp;

        child_prior_tmp.assign(child_prior.begin(), child_prior.end());
        qvalues = root->get_q(discount);
        csoftmax(child_prior_tmp, child_prior_tmp.size());
        // TODO: should be raw_value here
        float value = compute_mixed_value(root->raw_value, qvalues, child_visit, child_prior_tmp);
        std::vector<float> completed_qvalue;

        for (unsigned int i = 0;i < child_prior_tmp.size();i++){
            if (child_visit[i] > 0){
                completed_qvalue.push_back(qvalues[i]);
            }
            else{
                completed_qvalue.push_back(value);
            }
        }

        if (rescale_values){
            rescale_qvalues(completed_qvalue, epsilon);
        }

        float max_visit = *max_element(child_visit.begin(), child_visit.end());
        float visit_scale = maxvisit_init + max_visit;

        for (unsigned int i=0;i < completed_qvalue.size();i++){
            completed_qvalue[i] = completed_qvalue[i] * visit_scale * value_scale;
        }
        return completed_qvalue;
            
    }

    std::vector<int> get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations){
        std::vector<int> visit_seq;
        if(max_num_considered_actions <= 1){
            for (int i=0;i < num_simulations;i++)
                visit_seq.push_back(i);
            return visit_seq;
        }

        int log2max = std::ceil(std::log2(max_num_considered_actions));
        std::vector<int> visits;
        for (int i = 0;i < max_num_considered_actions;i++)
            visits.push_back(0);
        int num_considered = max_num_considered_actions;
        while (visit_seq.size() < num_simulations){
            int num_extra_visits = std::max(1, (int)(num_simulations / (log2max * num_considered)));
            for (int i = 0;i < num_extra_visits;i++){
                visit_seq.insert(visit_seq.end(), visits.begin(), visits.begin() + num_considered);
                for (int j = 0;j < num_considered;j++)
                    visits[j] += 1;
            }
            num_considered = std::max(2, num_considered/2);
        }
        std::vector<int> visit_seq_slice;
        visit_seq_slice.assign(visit_seq.begin(), visit_seq.begin() + num_simulations);
        return visit_seq_slice;
    }

    std::vector<std::vector<int>> get_table_of_considered_visits(int max_num_considered_actions, int num_simulations){
        std::vector<std::vector<int>> table;
        for (int m=0;m < max_num_considered_actions+1;m++){
            table.push_back(get_sequence_of_considered_visits(m, num_simulations));
        }
        return table;
    }

    std::vector<float> score_considered(int considered_visit, std::vector<float> gumbel, std::vector<float> logits, std::vector<float> normalized_qvalues, std::vector<int> visit_counts){
        float low_logit = -1e9;
        float max_logit = *max_element(logits.begin(), logits.end());
        for (unsigned int i=0;i < logits.size();i++){
            logits[i] -= max_logit;
        }
        std::vector<float> penalty;
        for (unsigned int i=0;i < visit_counts.size();i++){
            if (visit_counts[i]==considered_visit)
                penalty.push_back(0);
            else
                penalty.push_back(-std::numeric_limits<float>::infinity());
        }
        
        assert(gumbel.size()==logits.size()==normalized_qvalues.size()==penalty.size());
        std::vector<float> score;
        for (unsigned int i=0;i < visit_counts.size();i++){
            score.push_back(std::max(low_logit, gumbel[i] + logits[i] + normalized_qvalues[i]) + penalty[i]);
        }

        return score;
    }

    std::vector<float> generate_gumbel(float gumbel_scale, float gumbel_rng, int shape){
        std::mt19937 gen{gumbel_rng};
        std::extreme_value_distribution<float> d(0, 1);

        std::vector<float> gumbel;
        for (int i = 0;i < shape;i++)
            gumbel.push_back(gumbel_scale * d(gen));
        return gumbel;
    }

    /////////////////////////////

}