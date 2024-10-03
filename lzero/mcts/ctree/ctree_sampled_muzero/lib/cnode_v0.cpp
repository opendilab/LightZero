// C++11

#include <iostream>
#include "cnode.h"
#include <algorithm>
#include <map>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <stack>
#include <math.h>

#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <time.h>
#include <cassert>

#ifdef _WIN32
#include "..\..\common_lib\utils.cpp"
#else
#include "../../common_lib/utils.cpp"
#endif



template <class T>
size_t hash_combine(std::size_t &seed, const T &val)
{
    /*
    Overview:
        Combines a hash value with a new value using a bitwise XOR and a rotation.
        This function is used to create a hash value for multiple values.
    Arguments:
        - seed The current hash value to be combined with.
        - val The new value to be hashed and combined with the seed.
    */
    std::hash<T> hasher;  // Create a hash object for the new value.
    seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);  // Combine the new hash value with the seed.
    return seed;
}

// Sort by the value of second in descending order.
bool cmp(std::pair<int, double> x, std::pair<int, double> y)
{
    return x.second > y.second;
}

namespace tree
{
    //*********************************************************

    CAction::CAction()
    {
        /*
        Overview:
            Initialization of CAction. Parameterized constructor.
        */
        this->is_root_action = 0;
    }

    CAction::CAction(std::vector<float> value, int is_root_action)
    {
        /*
        Overview:
            Initialization of CAction with value and is_root_action. Default constructor.
        Arguments:
            - value: a multi-dimensional action.
            - is_root_action: whether value is a root node.
        */
        this->value = value;
        this->is_root_action = is_root_action;
    }

    CAction::~CAction() {} // Destructors.

    std::vector<size_t> CAction::get_hash(void)
    {
        /*
        Overview:
            get a hash value for each dimension in the multi-dimensional action.
        */
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
        /*
        Overview:
            Get the final combined hash value from the hash values of each dimension of the multi-dimensional action.
        */
        std::vector<size_t> hash = this->get_hash();
        size_t combined_hash = hash[0];

        if (hash.size() >= 1)
        {
            for (int i = 1; i < hash.size(); ++i)
            {
                combined_hash = hash_combine(combined_hash, hash[i]);
            }
        }

        return combined_hash;
    }

    //*********************************************************

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
        this->action_space_size = 9;
        this->num_of_sampled_actions = 20;
        this->continuous_action_space = false;

        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        CAction best_action;
        this->best_action = best_action;

        this->to_play = 0;
//        this->parent_value_prefix = 0.0;
        this->reward = 0.0;
    }

    CNode::CNode(float prior, std::vector<CAction> &legal_actions, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {
        /*
        Overview:
            Initialization of CNode with prior, legal actions, action_space_size, num_of_sampled_actions, continuous_action_space.
        Arguments:
            - prior: the prior value of this node.
            - legal_actions: a vector of legal actions of this node.
            - action_space_size: the size of action space of the current env.
            - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
            - continuous_action_space: whether the action space is continous in current env.
        */
        this->prior = prior;
        this->legal_actions = legal_actions;

        this->action_space_size = action_space_size;
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->continuous_action_space = continuous_action_space;
        this->is_reset = 0;
        this->visit_count = 0;
        this->value_sum = 0;
        this->to_play = 0;
//        this->value_prefix = 0.0;
//        this->parent_value_prefix = 0.0;
        this->current_latent_state_index = -1;
        this->batch_index = -1;
    }

    CNode::~CNode() {}


    void CNode::expand(int to_play, int current_latent_state_index, int batch_index, float reward, const std::vector<float> &policy_logits)
    {
        /*
        Overview:
            Expand the child nodes of the current node.
        Arguments:
            - to_play: which player to play the game in the current node.
            - current_latent_state_index: the x/first index of hidden state vector of the current node, i.e. the search depth.
            - batch_index: the y/second index of hidden state vector of the current node, i.e. the index of batch root node, its maximum is ``batch_size``/``env_num``.
            - reward: the reward of the current node.
            - policy_logits: the logit of the child nodes.
        */
        this->to_play = to_play;
        this->current_latent_state_index = current_latent_state_index;
        this->batch_index = batch_index;
        this->reward = reward;
        int action_num = policy_logits.size();

        #ifdef _WIN32
        // 创建动态数组
        float* policy = new float[action_num];
        #else
        float policy[action_num];
        #endif

        std::vector<int> all_actions;
        for (int i = 0; i < action_num; ++i)
        {
            all_actions.push_back(i);
        }
        std::vector<std::vector<float> > sampled_actions_after_tanh;
        std::vector<float> sampled_actions_log_probs_after_tanh;

        std::vector<int> sampled_actions;
        std::vector<float> sampled_actions_log_probs;
        std::vector<float> sampled_actions_probs;
        std::vector<float> probs;

        /*
        Overview:
            When the currennt env has continuous action space, sampled K actions from continuous gaussia distribution policy.
            When the currennt env has discrete action space, sampled K actions from discrete categirical distribution policy.

        */
        if (this->continuous_action_space == true)
        {
            // continuous action space for sampled algo..
            this->action_space_size = policy_logits.size() / 2;
            std::vector<float> mu;
            std::vector<float> sigma;
            for (int i = 0; i < this->action_space_size; ++i)
            {
                mu.push_back(policy_logits[i]);
                sigma.push_back(policy_logits[this->action_space_size + i]);
            }

            // The number of nanoseconds that have elapsed since epoch(1970: 00: 00 UTC on January 1, 1970). unsigned type will truncate this value.
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            // SAC-like tanh, pleasee refer to paper https://arxiv.org/abs/1801.01290.
            std::vector<std::vector<float> > sampled_actions_before_tanh;

            float sampled_action_one_dim_before_tanh;
            std::vector<float> sampled_actions_log_probs_before_tanh;

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
                    sampled_action_before_tanh.push_back(sampled_action_one_dim_before_tanh);
                    sampled_action_after_tanh.push_back(tanh(sampled_action_one_dim_before_tanh));
                    y.push_back(1 - pow(tanh(sampled_action_one_dim_before_tanh), 2) + 1e-6);
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
            // discrete action space for sampled algo..

            //========================================================
            // python code
            //========================================================
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

            //========================================================
            // TODO(pu): legal actions
            //========================================================
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

            // python code: legal_actions = []
            std::vector<CAction> legal_actions;

            // python code: probs = softmax(policy_logits)
            float logits_exp_sum = 0;
            for (int i = 0; i < policy_logits.size(); ++i)
            {
                logits_exp_sum += exp(policy_logits[i]);
            }
            for (int i = 0; i < policy_logits.size(); ++i)
            {
                probs.push_back(exp(policy_logits[i]) / (logits_exp_sum + 1e-6));
            }

            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

            // cout << "sampled_action[0]:" << sampled_action[0] <<endl;

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

            // Use the reciprocal of the probability value as the exponent and a random number sampled from a uniform distribution as the base:
            // Equivalent to adding a uniform random disturbance to the original probability value.
            for (auto prob : probs)
            {
                disturbed_probs.push_back(std::pow(uniform_distribution(generator), 1. / prob));
            }

            // Sort from large to small according to the probability value after the disturbance:
            // After sorting, the first vector is the index, and the second vector is the probability value after perturbation sorted from large to small.
            for (size_t iter = 0; iter < disturbed_probs.size(); iter++)
            {

            #ifdef __GNUC__
                // Use push_back for GCC
                disc_action_with_probs.push_back(std::make_pair(iter, disturbed_probs[iter]));
            #else
                // Use emplace_back for other compilers
                disc_action_with_probs.emplace_back(std::make_pair(iter, disturbed_probs[iter]));
            #endif
            }

            std::sort(disc_action_with_probs.begin(), disc_action_with_probs.end(), cmp);

            // take the fist ``num_of_sampled_actions`` actions
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

            // TODO(pu): fixed k, only for debugging
            //  Take the first ``num_of_sampled_actions`` actions: k=0,1,...,K-1
            // for (int k = 0; k < num_of_sampled_actions; ++k)
            // {
            //     sampled_actions.push_back(k);
            //     // disc_action_with_probs[k].second is disturbed_probs
            //     // sampled_actions_probs.push_back(disc_action_with_probs[k].second);
            //     sampled_actions_probs.push_back(probs[k]);
            // }

            disturbed_probs.clear();        // Empty the collection to prepare for the next sampling.
            disc_action_with_probs.clear(); // Empty the collection to prepare for the next sampling.
        }

        float prior;
        for (int i = 0; i < this->num_of_sampled_actions; ++i)
        {

            if (this->continuous_action_space == true)
            {
                CAction action = CAction(sampled_actions_after_tanh[i], 0);
                std::vector<CAction> legal_actions;
                this->children[action.get_combined_hash()] = CNode(sampled_actions_log_probs_after_tanh[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space); // only for muzero/efficient zero, not support alphazero
                this->legal_actions.push_back(action);
            }
            else
            {
                std::vector<float> sampled_action_tmp;
                for (size_t iter = 0; iter < 1; iter++)
                {
                    sampled_action_tmp.push_back(float(sampled_actions[i]));
                }
                CAction action = CAction(sampled_action_tmp, 0);
                std::vector<CAction> legal_actions;
                this->children[action.get_combined_hash()] = CNode(sampled_actions_probs[i], legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space); // only for muzero/efficient zero, not support alphazero
                this->legal_actions.push_back(action);
            }
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
        for (int i = 0; i < this->num_of_sampled_actions; ++i)
        {

            noise = noises[i];
            CNode *child = this->get_child(this->legal_actions[i]);
            prior = child->prior;
            if (this->continuous_action_space == true)
            {
                // if prior is log_prob
                child->prior = log(exp(prior) * (1 - exploration_fraction) + noise * exploration_fraction + 1e-6);
            }
            else
            {
                // if prior is prob
                child->prior = prior * (1 - exploration_fraction) + noise * exploration_fraction;
            }
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
//        float parent_value_prefix = this->value_prefix;
        for (auto a : this->legal_actions)
        {
            CNode *child = this->get_child(a);
            if (child->visit_count > 0)
            {
//                float true_reward = child->value_prefix - parent_value_prefix;
//                if (this->is_reset == 1)
//                {
//                    true_reward = child->value_prefix;
//                }
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

    std::vector<std::vector<float> > CNode::get_trajectory()
    {
        /*
        Overview:
            Find the current best trajectory starts from the current node.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from this node.
        */
        std::vector<CAction> traj;

        CNode *node = this;
        CAction best_action = node->best_action;
        while (best_action.is_root_action != 1)
        {
            traj.push_back(best_action);
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

    CNode *CNode::get_child(CAction action)
    {
        /*
        Overview:
            Get the child node corresponding to the input action.
        Arguments:
            - action: the action to get child.
        */
        return &(this->children[action.get_combined_hash()]);
        // TODO(pu): no hash
        // return &(this->children[action]);
        // return &(this->children[action.value[0]]);
    }

    //*********************************************************

    CRoots::CRoots()
    {
        this->root_num = 0;
        this->num_of_sampled_actions = 20;
    }

    CRoots::CRoots(int root_num, std::vector<std::vector<float> > legal_actions_list, int action_space_size, int num_of_sampled_actions, bool continuous_action_space)
    {
        /*
        Overview:
            Initialization of CNode with root_num, legal_actions_list, action_space_size, num_of_sampled_actions, continuous_action_space.
        Arguments:
            - root_num: the number of the current root.
            - legal_action_list: the vector of the legal action of this root.
            - action_space_size: the size of action space of the current env.
            - num_of_sampled_actions: the number of sampled actions, i.e. K in the Sampled MuZero papers.
            - continuous_action_space: whether the action space is continous in current env.
        */
        this->root_num = root_num;
        this->legal_actions_list = legal_actions_list;
        this->continuous_action_space = continuous_action_space;

        // sampled related core code
        this->num_of_sampled_actions = num_of_sampled_actions;
        this->action_space_size = action_space_size;

        for (int i = 0; i < this->root_num; ++i)
        {
            if (this->continuous_action_space == true and this->legal_actions_list[0][0] == -1)
            {
                // continous action space
                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }
            else if (this->continuous_action_space == false or this->legal_actions_list[0][0] == -1)
            {
                // sampled
                // discrete action space without action mask
                std::vector<CAction> legal_actions;
                this->roots.push_back(CNode(0, legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }

            else
            {
                // TODO(pu): discrete action space
                std::vector<CAction> c_legal_actions;
                for (int i = 0; i < this->legal_actions_list.size(); ++i)
                {
                    CAction c_legal_action = CAction(legal_actions_list[i], 0);
                    c_legal_actions.push_back(c_legal_action);
                }
                this->roots.push_back(CNode(0, c_legal_actions, this->action_space_size, this->num_of_sampled_actions, this->continuous_action_space));
            }
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

        // sampled related core code
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
        this->roots.clear();
    }

    std::vector<std::vector<std::vector<float> > > CRoots::get_trajectories()
    {
        /*
        Overview:
            Find the current best trajectory starts from each root.
        Returns:
            - traj: a vector of node index, which is the current best trajectory from each root.
        */
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

    // sampled related core code
    std::vector<std::vector<std::vector<float> > > CRoots::get_sampled_actions()
    {
        /*
        Overview:
            Get the sampled_actions of each root.
        Returns:
            - python_sampled_actions: a vector of sampled_actions for each root, e.g. the size of original action space is 6, the K=3, 
            python_sampled_actions = [[1,3,0], [2,4,0], [5,4,1]].
        */
        std::vector<std::vector<CAction> > sampled_actions;
        std::vector<std::vector<std::vector<float> > > python_sampled_actions;

        //  sampled_actions.reserve(this->root_num);

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
//                float parent_value_prefix = 0.0;
//                int is_reset = 0;
//                if (i >= 1)
//                {
//                    CNode *parent = search_path[i - 1];
//                    parent_value_prefix = parent->value_prefix;
//                    is_reset = parent->is_reset;
//                }
//
//                float true_reward = node->value_prefix - parent_value_prefix;
               min_max_stats.update(true_reward + discount_factor * node->value());

//                if (is_reset == 1)
//                {
//                    // parent is reset.
//                    true_reward = node->value_prefix;
//                }

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

                float true_reward = node->reward;

//                float parent_value_prefix = 0.0;
//                int is_reset = 0;
//                if (i >= 1)
//                {
//                    CNode *parent = search_path[i - 1];
//                    parent_value_prefix = parent->value_prefix;
//                    is_reset = parent->is_reset;
//                }
//
//                // NOTE: in self-play-mode, value_prefix is not calculated according to the perspective of current player of node,
//                // but treated as 1 player, just for obtaining the true reward in the perspective of current player of node.
//                float true_reward = node->value_prefix - parent_value_prefix;

                min_max_stats.update(true_reward + discount_factor * node->value());

//                if (is_reset == 1)
//                {
//                    // parent is reset.
//                    true_reward = node->value_prefix;
//                }
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
//            // reset
//            results.nodes[i]->is_reset = is_reset_list[i];

            cbackpropagate(results.search_paths[i], min_max_stats_lst->stats_lst[i], to_play_batch[i], values[i], discount_factor);
        }
    }

    CAction cselect_child(CNode *root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount_factor, float mean_q, int players, bool continuous_action_space)
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
            - continuous_action_space: whether the action space is continous in current env.
        Returns:
            - action: the action to select.
        */
        // sampled related core code
        // TODO(pu): Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
        float max_score = FLOAT_MIN;
        const float epsilon = 0.000001;
        std::vector<CAction> max_index_lst;
        for (auto a : root->legal_actions)
        {

            CNode *child = root->get_child(a);
            // sampled related core code
            float temp_score = cucb_score(root, child, min_max_stats, mean_q, root->visit_count - 1, pb_c_base, pb_c_init, discount_factor, players, continuous_action_space);

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

        // python code: int action = 0;
        CAction action;
        if (max_index_lst.size() > 0)
        {
            int rand_index = rand() % max_index_lst.size();
            action = max_index_lst[rand_index];
        }
        return action;
    }

    // sampled related core code
    float cucb_score(CNode *parent, CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, float total_children_visit_counts, float pb_c_base, float pb_c_init, float discount_factor, int players, bool continuous_action_space)
    {
        /*
        Overview:
            Compute the ucb score of the child.
        Arguments:
            - child: the child node to compute ucb score.
            - min_max_stats: a tool used to min-max normalize the score.
            - parent_mean_q: the mean q value of the parent node.
            - total_children_visit_counts: the total visit counts of the child nodes of the parent node.
            - pb_c_base: constants c2 in muzero.
            - pb_c_init: constants c1 in muzero.
            - disount_factor: the discount factor of reward.
            - players: the number of players.
            - continuous_action_space: whether the action space is continous in current env.
        Returns:
            - ucb_value: the ucb score of the child.
        */
        float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
        pb_c = log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= (sqrt(total_children_visit_counts) / (child->visit_count + 1));

        // prior_score = pb_c * child->prior;

        // sampled related core code
        // TODO(pu): empirical distribution
        std::string empirical_distribution_type = "density";
        if (empirical_distribution_type.compare("density"))
        {
            if (continuous_action_space == true)
            {
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += exp(parent->get_child(parent->legal_actions[i])->prior);
                }
                prior_score = pb_c * exp(child->prior) / (empirical_prob_sum + 1e-6);
            }
            else
            {
                float empirical_prob_sum = 0;
                for (int i = 0; i < parent->children.size(); ++i)
                {
                    empirical_prob_sum += parent->get_child(parent->legal_actions[i])->prior;
                }
                prior_score = pb_c * child->prior / (empirical_prob_sum + 1e-6);
            }
        }
        else if (empirical_distribution_type.compare("uniform"))
        {
            prior_score = pb_c * 1 / parent->children.size();
        }
        // sampled related core code
        if (child->visit_count == 0)
        {
            value_score = parent_mean_q;
        }
        else
        {
//            float true_reward = child->value_prefix - parent_value_prefix;
//            if (is_reset == 1)
//            {
//                true_reward = child->value_prefix;
//            }
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

    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount_factor, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> &virtual_to_play_batch, bool continuous_action_space)
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
            - continuous_action_space: whether the action space is continous in current env.
        */
        // set seed
        get_time_and_set_rand_seed();

//        int last_action = -1;

        std::vector<float> null_value;
        for (int i = 0; i < 1; ++i)
        {
            null_value.push_back(i + 0.1);
        }
        // CAction last_action = CAction(null_value, 1);
        std::vector<float> last_action;
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
                is_root = 0;
                parent_q = mean_q;

                CAction action = cselect_child(node, min_max_stats_lst->stats_lst[i], pb_c_base, pb_c_init, discount_factor, mean_q, players, continuous_action_space);
                if (players > 1)
                {
                    assert(virtual_to_play_batch[i] == 1 || virtual_to_play_batch[i] == 2);
                    if (virtual_to_play_batch[i] == 1)
                        virtual_to_play_batch[i] = 2;
                    else
                        virtual_to_play_batch[i] = 1;
                }

                node->best_action = action; // CAction
                // next
                node = node->get_child(action);
                last_action = action.value;

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

}
