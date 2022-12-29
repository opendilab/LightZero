#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <iostream>
#include <math.h>

template <class T>
size_t hash_combine(std::size_t &seed, const T &v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

//根据second的值降序排序
bool cmp(std::pair<int, double> x, std::pair<int, double> y)
{
    return x.second > y.second;
}

namespace tree
{
    //*********************************************************

    CAction::CAction()
    {
        this->is_root_action = 0;
    }

    CAction::CAction(std::vector<float> value, int is_root_action)
    {
        this->value = value;
        this->is_root_action = is_root_action;
    }

    CAction::~CAction() {}

    std::vector<size_t> CAction::get_hash(void)
    {
        std::vector<size_t> hash;
        for (int i = 0; i < this->value.size(); ++i)
        {
            std::size_t hash_i = std::hash<std::string>()(std::to_string(this->value[i]));
            hash.push_back(hash_i);
        }
        return hash;
    }
    size_t CAction::get_combined_hash(void)
    {
        std::vector<size_t> hash = this->get_hash();
        size_t combined_hash = hash[0];
        for (int i = 1; i < hash.size(); ++i)
        {
            combined_hash = hash_combine(combined_hash, hash[i]);
        }
        return combined_hash;
    }

    //*********************************************************

    CSearchResults::CSearchResults()
    {
        this->num = 0;
    }

    CSearchResults::CSearchResults(int num)
    {
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
        this->prior = 0;
        //        this->legal_actions = CAction(-1);
        //        this->legal_actions = -1;
        this->action_space_size = 9;
        this->num_of_sampled_actions = 20;
        this->continuous_action_space = false;


        //        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        //        this->best_action = -1;
        //        CAction best_action = CAction(-1);
        CAction best_action;
        this->best_action = best_action;

        this->to_play = 0;
        //        this->parent_value_prefix = 0.0;
    }

    //    CNode::CNode(float prior, std::vector<int> &legal_actions){
    CNode::CNode(float prior, std::vector<CAction> &legal_actions, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {

        this->prior = prior;
        this->legal_actions = legal_actions;

        this->action_space_size = action_space_size;
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->continuous_action_space = continuous_action_space;


        this->visit_count = 0;
        this->value_sum = 0;
        //        this->best_action = -1;
        this->to_play = 0;
        //        this->value_prefix = 0.0;
        //        this->parent_value_prefix = 0.0;
        this->hidden_state_index_x = -1;
        this->hidden_state_index_y = -1;
    }

    CNode::~CNode() {}

    void CNode::expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float reward, const std::vector<float> &policy_logits)
    {
        //         std::cout << "position 1" << std::endl;

        this->to_play = to_play;
        this->hidden_state_index_x = hidden_state_index_x;
        this->hidden_state_index_y = hidden_state_index_y;
        this->reward = reward;
        int action_num = policy_logits.size();
        // sampled related code
        std::vector<int> all_actions;
        for (int i = 0; i < action_num; ++i)
        {
            all_actions.push_back(i);
        }
        //        if(this->legal_actions.size()==0)
        //            for (int i = 0; i < 1; ++i){
        //              }
        //            this->legal_actions.assign(all_actions.begin(), all_actions.end());

        std::vector<std::vector<float> > sampled_actions_after_tanh;
        std::vector<float> sampled_actions_log_probs_after_tanh;

        std::vector<int> sampled_actions;
        std::vector<float> sampled_actions_log_probs;
        std::vector<float> sampled_actions_probs;
        // std::cout << "position sampled_actions_probs;" << std::endl;

        // sampled from gaussia distribution in continuous action space
        // sampled from categirical distribution in discrete action space
        if (this->continuous_action_space == true)
        {
            // continuous action space for sampled ez
            this->action_space_size = policy_logits.size() / 2;
            std::vector<float> mu;
            std::vector<float> sigma;
            for (int i = 0; i < this->action_space_size; ++i)
            {
                mu.push_back(policy_logits[i]);
                sigma.push_back(policy_logits[this->action_space_size + i]);
            }

            // 从epoch（1970年1月1日00:00:00 UTC）开始经过的纳秒数，unsigned类型会截断这个值
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            // way 1: no tanh
            //        std::vector<std::vector<float> > sampled_actions;
            //        float sampled_action_one_dim;
            //        std::vector<float> sampled_actions_log_probs;
            //        std::default_random_engine generator(seed);
            // 第一个参数为高斯分布的平均值，第二个参数为标准差
            //      std::normal_distribution<double> distribution(mu, sigma);

            //        for (int i = 0; i < this->num_of_sampled_actions; ++i){
            ////            std::cout << "-------------------------" <<std::endl;
            ////            std::cout << "num_of_sampled_actions index:" << i <<std::endl;
            ////            std::cout << "-------------------------" <<std::endl;
            //            float sampled_action_prob = 1;
            //            // TODO(pu): why here
            //            std::vector<float> sampled_action;
            //
            //            for (int j = 0; j < this->action_space_size; ++j){
            ////                 std::cout << "sampled_action_prob: " << sampled_action_prob <<std::endl;
            ////                 std::cout << "mu[j], sigma[j]: " << mu[j] << sigma[j]<<std::endl;
            //
            //                std::normal_distribution<float> distribution(mu[j], sigma[j]);
            //                sampled_action_one_dim = distribution(generator);
            //                // refer to python normal log_prob method
            //                sampled_action_prob *= exp(- pow((sampled_action_one_dim - mu[j]), 2) / (2 * pow(sigma[j], 2)) -  log(sigma[j]) - log(sqrt(2 * M_PI)));
            ////                std::cout << "sampled_action_one_dim:" << sampled_action_one_dim <<std::endl;
            //
            //                sampled_action.push_back(sampled_action_one_dim);
            //
            //            }
            //            sampled_actions.push_back(sampled_action);
            //            sampled_actions_log_probs.push_back(log(sampled_action_prob));
            //           }

            // way 2: sac-like tanh
            std::vector<std::vector<float> > sampled_actions_before_tanh;
            // std::vector<std::vector<float> > sampled_actions_after_tanh;

            float sampled_action_one_dim_before_tanh;
            std::vector<float> sampled_actions_log_probs_before_tanh;
            // std::vector<float> sampled_actions_log_probs_after_tanh;

            std::default_random_engine generator(seed);
            for (int i = 0; i < this->num_of_sampled_actions; ++i)
            {
                float sampled_action_prob_before_tanh = 1;
                // TODO(pu): why here
                std::vector<float> sampled_action_before_tanh;
                std::vector<float> sampled_action_after_tanh;
                std::vector<float> y;

                for (int j = 0; j < this->action_space_size; ++j)
                {
                    std::normal_distribution<float> distribution(mu[j], sigma[j]);
                    sampled_action_one_dim_before_tanh = distribution(generator);
                    // refer to python normal log_prob method
                    sampled_action_prob_before_tanh *= exp(-pow((sampled_action_one_dim_before_tanh - mu[j]), 2) / (2 * pow(sigma[j], 2)) - log(sigma[j]) - log(sqrt(2 * M_PI)));
                    //                std::cout << "sampled_action_one_dim:" << sampled_action_one_dim <<std::endl;
                    sampled_action_before_tanh.push_back(sampled_action_one_dim_before_tanh);
                    sampled_action_after_tanh.push_back(tanh(sampled_action_one_dim_before_tanh));
                    // y = 1 - pow(sampled_actions, 2) + 1e-9;
                    y.push_back(1 - pow(tanh(sampled_action_one_dim_before_tanh), 2) + 1e-9);
                }
                sampled_actions_before_tanh.push_back(sampled_action_before_tanh);
                sampled_actions_after_tanh.push_back(sampled_action_after_tanh);
                sampled_actions_log_probs_before_tanh.push_back(log(sampled_action_prob_before_tanh));
                float y_sum = std::accumulate(y.begin(), y.end(), 0.);
                sampled_actions_log_probs_after_tanh.push_back(log(sampled_action_prob_before_tanh) - log(y_sum));
            }
        }
        else
        {
            // discrete action space for sampled ez

            //###############
            // python version code
            //###############
            // if self.legal_actions is not None:
            //     # fisrt use the self.legal_actions to exclude the illegal actions
            //     policy_tmp = [0. for _ in range(self.action_space_size)]
            //     for index, legal_action in enumerate(self.legal_actions):
            //         policy_tmp[legal_action] = policy_logits[index]
            //     policy_logits = policy_tmp
            // # then empty the self.legal_actions
            // self.legal_actions = []
            // then empty the self.legal_actions
            //            prob = torch.softmax(torch.tensor(policy_logits), dim=-1)
            //            sampled_actions = torch.multinomial(prob, self.num_of_sampled_actions, replacement=False)

            //###############
            // TODO(pu): legal actions
            //###############
            // std::vector<float> policy_tmp;
            // for (int i = 0; i < this->action_space_size; ++i)
            // {
            //     policy_tmp.push_back(0.);
            // }
            // for (int i = 0; i < this->legal_actions.size(); ++i)
            // {
            //     policy_tmp[this->legal_actions[i].value] = policy_logits[i];
            // }
            // for (int i = 0; i < this->action_space_size; ++i)
            // {
            //     policy_logits[i] = policy_tmp[i];
            // }
            // std::cout << "position 3" << std::endl;

            // python version code: legal_actions = []
            std::vector<CAction> legal_actions;
            std::vector<float> probs;

            // probs = softmax(policy_logits)
            float logits_exp_sum = 0;
            for (int i = 0; i < policy_logits.size(); ++i)
            {
                logits_exp_sum += exp(policy_logits[i]);
            }
            for (int i = 0; i < policy_logits.size(); ++i)
            {
                probs.push_back(exp(policy_logits[i]) / (logits_exp_sum + 1e-9));
            }

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            //    cout << "sampled_action[0]:" << sampled_action[0] <<endl;

            // std::vector<int> sampled_actions;
            // std::vector<float> sampled_actions_log_probs;
            // std::vector<float> sampled_actions_probs;
            std::default_random_engine generator(seed);

            //  有放回抽样
            // for (int i = 0; i < num_of_sampled_actions; ++i)
            // {
            //     float sampled_action_prob = 1;
            //     int sampled_action;

            //     std::discrete_distribution<float> distribution(probs.begin(), probs.end());

            //     // for (float x:distribution.probabilities()) std::cout << x << " ";
            //     sampled_action = distribution(generator);
            //     // std::cout << "sampled_action： " << sampled_action << std::endl;

            //     sampled_actions.push_back(sampled_action);
            //     sampled_actions_probs.push_back(probs[sampled_action]);
            //     std::cout << "sampled_actions_probs" << '[' << i << ']' << sampled_actions_probs[i] << std::endl;

            //     sampled_actions_log_probs.push_back(log(probs[sampled_action]));
            //     std::cout << "sampled_actions_log_probs" << '[' << i << ']' << sampled_actions_log_probs[i] << std::endl;
            // }

            // 每个节点的legal_actions应该为一个固定离散集合，所以采用无放回抽样
            // std::cout << "position uniform_distribution init" << std::endl;
            std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0); //均匀分布
            // std::cout << "position uniform_distribution done" << std::endl;
            std::vector<double> disturbed_probs;
            std::vector<std::pair<int, double> > disc_action_with_probs;

            // 把概率值作为指数，从均匀分布采样的随机数作为底数：
            // 相当于给原始概率值增加了均匀的随机扰动
            for (auto prob : probs)
            {
                disturbed_probs.push_back(std::pow(uniform_distribution(generator), 1. / prob));
            }

            // std::cout << "position 4" << std::endl;

            // 按照扰动后的概率值从大到小排序:
            // 排序后第一个向量为索引，第二个向量为从大到小排序的扰动后的概率值
            for (size_t iter = 0; iter < disturbed_probs.size(); iter++)
            {
                disc_action_with_probs.__emplace_back(std::make_pair(iter, disturbed_probs[iter]));
                // disc_action_with_probs.emplace_back(std::make_pair(iter, disturbed_probs[iter]));
            }
            // std::sort(disc_action_with_probs.begin(), disc_action_with_probs.end(), [](auto x, auto y)
            //           { return x.second > y.second; });

            std::sort(disc_action_with_probs.begin(), disc_action_with_probs.end(), cmp);

            // for (int j = 0; j < policy_logits.size(); ++j)
            // {
            //     std::cout << "disc_action_with_probs[i].first " << disc_action_with_probs[j].first << " "
            //               << "disc_action_with_probs[j].second " << disc_action_with_probs[j].second << std::endl;
            // }

            // 取前num_of_sampled_actions个动作
            for (int k = 0; k < num_of_sampled_actions; ++k)
            {
                sampled_actions.push_back(disc_action_with_probs[k].first);
                // disc_action_with_probs[k].second is disturbed_probs
                // sampled_actions_probs.push_back(disc_action_with_probs[k].second);
                sampled_actions_probs.push_back(probs[disc_action_with_probs[k].first]);


            // TODO(pu): logging
            // std::cout << "sampled_actions[k]： " << sampled_actions[k] << std::endl;
            // std::cout << "sampled_actions_probs[k]： " << sampled_actions_probs[k] << std::endl;
            }
            disturbed_probs.clear();        // 清空集合，为下次抽样做准备
            disc_action_with_probs.clear(); // 清空集合，为下次抽样做准备
            // std::cout << "position sampled doe" << std::endl;
        }

        float prior;
        for (int i = 0; i < this->num_of_sampled_actions; ++i)
        {

            if (this->continuous_action_space == true)
            {
                // std::cout << "position this->continuous_action_space == true done" << std::endl;
                //            prior = policy[a] / policy_sum;
                CAction action = CAction(sampled_actions_after_tanh[i], 0);
                // std::cout << "position done: CAction action = CAction(sampled_actions_after_tanh[i], 0);" << std::endl;
                std::vector<CAction> legal_actions;
                // backup: segment fault
                //            std::cout << "legal_actions[0]: " << legal_actions[0] << std::endl;
                //            std::cout << "legal_actions[0].is_root_action: " << legal_actions[0].is_root_action << std::endl;
                //            std::cout << "legal_actions.get_combined_hash()" << legal_actions[0].get_combined_hash() << std::endl;
                //            std::cout << "legal_actions.is_root_action: " << legal_actions[1].is_root_action << std::endl;

                //            this->children[action] = CNode(sampled_actions_log_prob[i], legal_actions, this->action_space_size, this->num_of_sampled_actions); // only for muzero/efficient zero, not support alphazero
                //            std::cout << "position 6" << std::endl;
                //            std::cout << "action.get_combined_hash()" << action.get_combined_hash() << std::endl;
                
                // std::cout << "position 8" << std::endl;

                // cpp debug
                // std::cout << "action.get_combined_hash()" << action.get_combined_hash() << std::endl;
                // std::cout << " sampled_actions_log_probs_after_tanh[i]: " << sampled_actions_log_probs_after_tanh[i] << std::endl;
                // std::cout << " this->action_space_size:  " << this->action_space_size << std::endl;
                // std::cout << " this->num_of_sampled_actions:  " << this->num_of_sampled_actions << std::endl;
                // std::cout << " this->continuous_action_space :  " << this->continuous_action_space  << std::endl;
                // CNode(sampled_actions_log_probs_after_tanh[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space);
                // cpp debug

                this->children[action.get_combined_hash()] = CNode(sampled_actions_log_probs_after_tanh[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space); // only for muzero/efficient zero, not support alphazero
                this->legal_actions.push_back(action);
            }
            else
            {
                std::vector<float> sampled_action_tmp;
                for (size_t iter = 0; iter < 1; iter++)
                {
                    // std::cout << "sampled_actions[i]: " << sampled_actions[i] <<std::endl;

                    sampled_action_tmp.push_back(float(sampled_actions[i]));
                    // std::cout << "float(sampled_actions[i]): " << float(sampled_actions[i]) <<std::endl;


                }
                // float sampled_action_tmp = static_cast<float>(sampled_actions[i]);
                CAction action = CAction(sampled_action_tmp, 0);

                // CAction action = CAction(static_cast<float>(sampled_actions[i]), 0);
                std::vector<CAction> legal_actions;
                // cpp debug
                // std::cout << "position 8" << std::endl;
                // std::cout << "action.get_combined_hash()" << action.get_combined_hash() << std::endl;
                // std::cout << " this->action_space_size:  " << this->action_space_size << std::endl;
                // std::cout << " this->num_of_sampled_actions:  " << this->num_of_sampled_actions << std::endl;
                // std::cout << " this->continuous_action_space :  " << this->continuous_action_space  << std::endl;
                // CNode(sampled_actions_probs[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space);
                // cpp debug

                // NOTE: 
                this->children[action.get_combined_hash()] = CNode(sampled_actions_probs[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space); // only for muzero/efficient zero, not support alphazero
                this->legal_actions.push_back(action);
            }
            // std::cout << "position 7" << std::endl;
            // for (int j = 0; j < this->action_space_size; ++j)
            // {
            //                std::cout << "action.value[j]： " << action.value[j] << std::endl;
            // }
        }


    }

    void CNode::add_exploration_noise(float exploration_fraction, const std::vector<float> &noises)
    {
        float noise, prior;
        //        for(auto a: this->legal_actions){
        //            noise = noises[a];
        //            CNode* child = this->get_child(a);
        for (int i = 0; i < this->legal_actions.size(); ++i)
        {
            noise = noises[i];
            CNode *child = this->get_child(this->legal_actions[i]);

            prior = child->prior;
            if (this->continuous_action_space == true){
                // prior is log_prob
                child->prior = log(exp(prior) * (1 - exploration_fraction) + noise * exploration_fraction + 1e-9);
            }
            else{
                // prior is prob
                child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
            }
        }
    }

    float CNode::get_mean_q(int isRoot, float parent_q, float discount)
    {
        float total_unsigned_q = 0.0;
        int total_visits = 0;
        //        float parent_value_prefix = this->value_prefix;
        for (auto a : this->legal_actions)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
                //                float true_reward = child->value_prefix - parent_value_prefix;
                float true_reward = child->reward;
                float qsa = true_reward + discount * child->value();
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
        return this->children.size() > 0;
    }

    float CNode::value()
    {
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

    //    std::vector<int> CNode::get_trajectory(){
    //        std::vector<int> traj;
    //
    //        CNode* node = this;
    //        int best_action = node->best_action;
    //        while(best_action >= 0){
    //            traj.push_back(best_action);
    //
    //            node = node->get_child(best_action);
    //            best_action = node->best_action;
    //        }
    //        return traj;
    //    }

    std::vector<std::vector<float> > CNode::get_trajectory()
    {
        std::vector<CAction> traj;

        CNode *node = this;
        CAction best_action = node->best_action;
        while (best_action.is_root_action != 1)
        {
            traj.push_back(best_action);
            //            best_action = CAction(best_action);
            node = node->get_child(best_action);
            best_action = node->best_action;
        }

        std::vector<std::vector<float> > traj_return;
        for (int i = 0; i < traj.size(); ++i)
        {
            traj_return.push_back(traj[i].value);
        }
        return traj_return;
    }

    std::vector<int> CNode::get_children_distribution()
    {
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

    //    CNode* CNode::get_child(int action){
    //        return &(this->children[action]);
    //    }
    CNode *CNode::get_child(CAction action)
    {
        //        return &(this->children[action]);
        return &(this->children[action.get_combined_hash()]);
    }

    //*********************************************************

    CRoots::CRoots()
    {
        this->root_num = 0;
        this->pool_size = 0;
    }

    //    CRoots::CRoots(int root_num, std::vector<std::vector<int> > &legal_actions_list, int action_space_size, int num_of_sampled_actions){
    CRoots::CRoots(int root_num, std::vector<std::vector<float> > legal_actions_list, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {

        this->root_num = root_num;
        //        this->pool_size = pool_size;
        this->legal_actions_list = legal_actions_list;
        this->continuous_action_space = continuous_action_space;

        // sampled related code
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->action_space_size = action_space_size;

        // std::cout << "position here" << std::endl;


        for (int i = 0; i < this->root_num; ++i)
        {
            //            this->roots.push_back(CNode(0, this->legal_actions_list[i], this->num_of_sampled_actions));
            if (this->continuous_action_space == true and this->legal_actions_list[0][0] == -1)
            {
                //  continous action space
                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
                // std::cout << "position -110" << std::endl;

            }
            else if (this->continuous_action_space == false or this->legal_actions_list[0][0] == -1)
            {
                //  sampled
                // discrete action space without action mask

                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));

            }

            else
            {
                // TODO(pu): discrete action space

                //                    this->roots.push_back(CNode(0, this->legal_actions_list[i], this->action_space_size, this->num_of_sampled_actions));
                std::vector<CAction> c_legal_actions;
                for (int i = 0; i < this->legal_actions_list.size(); ++i)
                {
                    CAction c_legal_action = CAction(legal_actions_list[i], 0);
                    c_legal_actions.push_back(c_legal_action);
                }
                //                  std::cout << "action_space_size:" << action_space_size << std::endl;
                //                  std::cout << "num_of_sampled_actions:" << num_of_sampled_actions <<std::endl;
                this->roots.push_back(CNode(0, c_legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));

            }
            // TODO
            //            std::vector<std::vector<float> > legal_actions;
        }
    }

    CRoots::~CRoots() {}

    void CRoots::prepare(float root_exploration_fraction, const std::vector<std::vector<float> > &noises, const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i]);
            this->roots[i].add_exploration_noise(root_exploration_fraction, noises[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::prepare_no_noise(const std::vector<float> &rewards, const std::vector<std::vector<float> > &policies, std::vector<int> &to_play_batch)
    {
        for (int i = 0; i < this->root_num; ++i)
        {
            this->roots[i].expand(to_play_batch[i], 0, i, rewards[i], policies[i]);

            this->roots[i].visit_count += 1;
        }
    }

    void CRoots::clear()
    {
        this->roots.clear();
    }

    std::vector<std::vector<std::vector<float> > > CRoots::get_trajectories()
    {

        std::vector<std::vector<std::vector<float> > > trajs;
        trajs.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            trajs.push_back(this->roots[i].get_trajectory());
        }
        return trajs;
    }

    std::vector<std::vector<int> > CRoots::get_distributions()
    {
        std::vector<std::vector<int> > distributions;
        distributions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            distributions.push_back(this->roots[i].get_children_distribution());
        }
        return distributions;
    }

    // sampled related code
    std::vector<std::vector<std::vector<float> > > CRoots::get_sampled_actions()
    {
        std::vector<std::vector<CAction> > sampled_actions;
        std::vector<std::vector<std::vector<float> > > python_sampled_actions;

        //        sampled_actions.reserve(this->root_num);

        for (int i = 0; i < this->root_num; ++i)
        {
            std::vector<CAction> sampled_action;
            sampled_action = this->roots[i].legal_actions;
            std::vector<std::vector<float> > python_sampled_action;

            for (int j = 0; j < this->roots[i].legal_actions.size(); ++j)
            {
                python_sampled_action.push_back(sampled_action[j].value);
            }
            python_sampled_actions.push_back(python_sampled_action);
        }

        return python_sampled_actions;
    }

    std::vector<float> CRoots::get_values()
    {
        std::vector<float> values;
        for (int i = 0; i < this->root_num; ++i)
        {
            values.push_back(this->roots[i].value());
        }
        return values;
    }

    //*********************************************************
    //
    void update_tree_q(CNode *root, tools::CMinMaxStats &min_max_stats, float discount, int players)
    {
        std::stack<CNode *> node_stack;
        node_stack.push(root);
        //        float parent_value_prefix = 0.0;
        while (node_stack.size() > 0)
        {
            CNode *node = node_stack.top();
            node_stack.pop();

            if (node != root)
            {
                //                # NOTE: in 2 player mode, value_prefix is not calculated according to the perspective of current player of node,
                //                # but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                //                # true_reward = node.value_prefix - (- parent_value_prefix)
                //                float true_reward = node->value_prefix - node->parent_value_prefix;
                float true_reward = node->reward;

                float qsa;
                if (players == 1)
                    qsa = true_reward + discount * node->value();
                else if (players == 2)
                    // TODO(pu):
                    qsa = true_reward + discount * (-1) * node->value();

                min_max_stats.update(qsa);
            }

            for (auto a : node->legal_actions)
            {
                CNode *child = node->get_child(a);
                if (child->expanded())
                {
                    //                    child->parent_value_prefix = node->value_prefix;
                    node_stack.push(child);
                }
            }
        }
    }

    void cback_propagate(std::vector<CNode *> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount)
    {
        if (to_play == 0)
        {
            float bootstrap_value = value;
            int path_len = search_path.size();
            for (int i = path_len - 1; i >= 0; --i)
            {
                CNode *node = search_path[i];
                node->value_sum += bootstrap_value;
                node->visit_count += 1;

                float true_reward = node->reward;

                min_max_stats.update(true_reward + discount * node->value());

                bootstrap_value = true_reward + discount * bootstrap_value;
            }
        }
        else
        {
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

                // NOTE: in 2 player mode, value_prefix is not calculated according to the perspective of current player of node,
                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
                //                float true_reward = node->value_prefix - parent_value_prefix;
                float true_reward = node->reward;

                // TODO(pu): why in muzero-general is - node.value
                min_max_stats.update(true_reward + discount * -node->value());

                if (node->to_play == to_play)
                    bootstrap_value = -true_reward + discount * bootstrap_value;
                else
                    bootstrap_value = true_reward + discount * bootstrap_value;
            }
        }
    }

    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float> > &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &to_play_batch)
    {
        for (int i = 0; i < results.num; ++i)
        {
            results.nodes[i]->expand(to_play_batch[i], hidden_state_index_x, i, value_prefixs[i], policies[i]);
            cback_propagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount);
        }
    }

    //    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players){
    CAction cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, int players, bool continuous_action_space)
    {

        // sampled related code
        // TODO(pu): Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<CAction> max_index_lst;
        for (auto a : root->legal_actions)
        {

            CNode *child = root->get_child(a);
            // sampled related code
            //            float temp_score = cucb_score(child, min_max_stats, mean_q, root->is_reset, root->visit_count - 1, root->value_prefix, pb_c_base, pb_c_init, discount, players);

            float temp_score = cucb_score(root, child, min_max_stats, mean_q, root->visit_count - 1, root->reward, pb_c_base, pb_c_init, discount, players, continuous_action_space);

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

        //        int action = 0;
        CAction action;
        if (max_index_lst.size() > 0)
        {
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
        //        return &action;
    }

    float cucb_score(CNode *parent, CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float parent_reward, float pb_c_base, float pb_c_init, float discount, int players, bool continuous_action_space)
    {
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        //        prior_score = pb_c * child->prior;

        // sampled related code
        // TODO(pu): empirical distribution
        std::string empirical_distribution_type = "density";
        if (empirical_distribution_type.compare("density"))
        {
            if (continuous_action_space == true)
            {
                // std::cout << "position float empirical_prob_sum = 0;" << std::endl;
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += exp(parent->get_child(parent->legal_actions[i])->prior);
                }
                prior_score = pb_c * exp(child->prior) / (empirical_prob_sum + 1e-9);
                // std::cout << "position prior_score = pb_c * exp(child->prior) / (empirical_prob_sum + 1e-9);" << std::endl;
            }
            else
            {
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += parent->get_child(parent->legal_actions[i])->prior;
                }
                prior_score = pb_c * child->prior / (empirical_prob_sum + 1e-9);
            }
        }
        else if (empirical_distribution_type.compare("uniform"))
        {
            prior_score = pb_c * 1 / parent->children.size();
        }
        // sampled related code

        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
            float true_reward = child->reward;
            if (players == 1)
                value_score = true_reward + discount * child->value();
            else if (players == 2)
                value_score = true_reward + discount * (-child->value());
        }

        value_score = min_max_stats.normalize(value_score);

        if (value_score < 0)
            value_score = 0;
        if (value_score > 1)
            value_score = 1;

        float ucb_value = prior_score + value_score;
        return ucb_value;
    }

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, bool continuous_action_space)
    {
        // set seed
        timeval t1;
        gettimeofday(&t1, NULL);
        srand(t1.tv_usec);

        //        int last_action = -1;

        std::vector<float> null_value;
        for (int i = 0; i < 1; ++i)
        {
            null_value.push_back(i + 0.1);
        }
        //        CAction last_action = CAction(null_value, 1);
        std::vector<float> last_action;

        float parent_q = 0.0;
        results.search_lens = std::vector<int>();

        int players = 0;
        int largest_element = *max_element(virtual_to_play_batch.begin(), virtual_to_play_batch.end()); // 0 or 2
        if (largest_element == 0)
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
                float mean_q = node->get_mean_q(is_root, parent_q, discount);
                is_root = 0;
                parent_q = mean_q;

                //                int action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players);
                CAction action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players, continuous_action_space);

                if (players > 1)
                {
                    if (virtual_to_play_batch[i] == 1)
                        virtual_to_play_batch[i] = 2;
                    else
                        virtual_to_play_batch[i] = 1;
                }

                node->best_action = action; // CAction

                // next
                node = node->get_child(action);
                //                last_action = action;
                last_action = action.value;

                results.search_paths[i].push_back(node);
                search_len += 1;
            }

            CNode *parent = results.search_paths[i][results.search_paths[i].size() - 2];

            results.hidden_state_index_x_lst.push_back(parent->hidden_state_index_x);
            results.hidden_state_index_y_lst.push_back(parent->hidden_state_index_y);

            results.last_actions.push_back(last_action);
            results.search_lens.push_back(search_len);
            results.nodes.push_back(node);
            results.virtual_to_play_batchs.push_back(virtual_to_play_batch[i]);
        }
    }

}
