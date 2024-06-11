// This code is a Python extension implemented in C++ using the pybind11 library.
// It's a Monte Carlo Tree Search (MCTS) algorithm with modifications based on Google's AlphaZero paper.
// MCTS is an algorithm for making optimal decisions in a certain class of combinatorial problems.
// It's most famously used in board games like chess, Go, and shogi.

// The following lines include the necessary headers to facilitate the implementation of the MCTS algorithm.

#include "node_gumbel_alphazero.h"
#include <cmath>
#include <map>
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

// This line creates an alias for the pybind11 namespace, making it easier to reference in the code.
namespace py = pybind11;

// This part defines the MCTS class and its member variables.
// The MCTS class implements the MCTS algorithm, and its member variables store configuration values used in the algorithm.
class MCTS {
    int max_moves;
    int num_simulations;
    double pb_c_base;
    double pb_c_init;
    double root_dirichlet_alpha;
    double root_noise_weight;
    int maxvisit_init;
    float value_scale;
    float gumbel_scale;
    float gumbel_rng;
    int max_num_considered_actions;
    std::vector<float> gumbel;
    py::object simulate_env;

// This part defines the constructor of the MCTS class.
// The constructor initializes the member variables with the provided arguments or with their default values.
public:
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25,
         int maxvisit_init=50, float value_scale=0.1,
         float gumbel_scale = 10.0, float gumbel_rng = 0.0,
         int max_num_considered_actions = 4, // parameters for gumbel alphazero
         py::object simulate_env=py::none())
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight),
          maxvisit_init(maxvisit_init), value_scale(value_scale),
          gumbel_scale(gumbel_scale), gumbel_rng(gumbel_rng),
          max_num_considered_actions(max_num_considered_actions),  // parameters for gumbel alphazero
          gumbel(_generate_gumbel(gumbel_scale, gumbel_rng, 36)),  //simulate_env.attr("action_space").attr("n").cast<int>())),
          simulate_env(simulate_env) {}

    // Methods: get_next_action，_simulate，_select_child，_expand_leaf_node，_ucb_score，_add_exploration_noise
    
    // This function calculates the Upper Confidence Bound (UCB) score for a given node in the MCTS tree based on the parent node's visit count,
    // the child node's visit count, and the child node's prior probability.
    double _ucb_score(Node* parent, Node* child) {
        double pb_c = std::log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(parent->visit_count) / (child->visit_count + 1);

        double prior_score = pb_c * child->prior_p;
        double value_score = child->get_value();
        return prior_score + value_score;
    }
    
    // This function adds Dirichlet noise to the prior probabilities of the actions of a given node to encourage exploration.
    void _add_exploration_noise(Node* node) {
    std::vector<int> actions;
    for (const auto& kv : node->children) {
        actions.push_back(kv.first);
    }

    std::default_random_engine generator;
    std::gamma_distribution<double> distribution(root_dirichlet_alpha, 1.0);

    std::vector<double> noise;
    double sum = 0;
    for (size_t i = 0; i < actions.size(); ++i) {
        double sample = distribution(generator);
        noise.push_back(sample);
        sum += sample;
    }

    // Normalize the samples to simulate a Dirichlet distribution
    for (size_t i = 0; i < noise.size(); ++i) {
        noise[i] /= sum;
    }

    double frac = root_noise_weight;
    for (size_t i = 0; i < actions.size(); ++i) {
        node->children[actions[i]]->prior_p = node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
    }
}
    // This function generates Gumbel noise for the MCTS algorithm.
    std::vector<float> _generate_gumbel(float gumbel_scale, float gumbel_rng, int shape) {
        std::vector<float> gumbel;
        std::mt19937 gen(static_cast<unsigned int>(gumbel_rng));
        std::extreme_value_distribution<float> dis(0, 1);
        for (int i = 0; i < shape; i++) {
            gumbel.push_back(gumbel_scale * dis(gen));
        }
        return gumbel;
    }

    // This function selects the child of a given node that has the highest UCB score among the legal actions.
    std::pair<int, Node*> _select_child(Node* node, py::object simulate_env) {
        int action = -1;
        Node* child = nullptr;
        double best_score = -9999999;
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            Node* child_tmp = kv.second;

            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();

            std::vector<int> legal_actions;
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }

            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                double score = _ucb_score(node, child_tmp);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    child = child_tmp;
                }
            }

        }
        if (child == nullptr) {
            child = node;
        }
        return std::make_pair(action, child);
    }

    // Gumbel related code
    // add different children node select function
    std::pair<int, Node*> _select_root_child(Node* node, py::object simulate_env) {
        std::vector<Node*> child_tmp_list;
        std::vector<int> action_list;
        std::vector<int> visit_count_list;

        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            Node* child_tmp = kv.second;
            child_tmp_list.push_back(child_tmp);
            action_list.push_back(action_tmp);
            visit_count_list.push_back(child_tmp->visit_count);
        }

        // get mixed q value of child nodes
        std::vector<float> completed_qvalues = _qtransform_completed_by_mix_value(node, child_tmp_list);
        // get table of considered_visits
        std::vector<std::vector<int> > table_of_considered_visits = get_table_of_considered_visits(max_num_considered_actions, num_simulations);

        // get number of actions
        int num_actions = action_list.size();
        int num_considered = std::min(max_num_considered_actions, num_simulations);
        // get the sum of visit counts of each action
        int simulation_index = std::accumulate(visit_count_list.begin(), visit_count_list.end(), 0);
        int considered_visit = table_of_considered_visits[num_considered][simulation_index];

        std::vector<float> score_considered = _score_considered(considered_visit, child_tmp_list, completed_qvalues);
        float argmax = -std::numeric_limits<float>::infinity();
        int action = -1;
        for (int i = 0; i < num_actions; i++) {
            if (score_considered[i] > argmax) {
                argmax = score_considered[i];
                action = action_list[i];
            }
        }

        return std::make_pair(action, node->children[action]);
    }

    // select interior child
    std::pair<int, Node*> _select_interior_child(Node* node, py::object simulate_env){
        // get visit_count and prior for each node
        std::vector<int> visit_counts;
        std::vector<float> priors;
        std::vector<Node*> children;
        for (const auto& kv : node->children) {
            visit_counts.push_back(kv.second->visit_count);
            priors.push_back(kv.second->prior_p);
            children.push_back(kv.second);
        }
        // get completed value
        std::vector<float> completed_value = _qtransform_completed_by_mix_value(node, children);
        // get probs
        std::vector<double> probs;
        for (int i = 0; i < visit_counts.size(); i++) {
            probs.push_back(priors[i] + completed_value[i]);
        }
        // softmax probs
        std::vector<double> probs_softmax = softmax(probs, 1);
        // calculate sum of visit count
        int sum_visit_count = std::accumulate(visit_counts.begin(), visit_counts.end(), 0);
        // get to_argmax
        std::vector<float> to_argmax;
        for (int i = 0; i < visit_counts.size(); i++) {
            to_argmax.push_back(probs[i] - (float)visit_counts[i]/(float)(1+sum_visit_count));
        }

        // Check if node->children is empty
        if (node->children.empty()) {
            // Return default value or throw exception as needed
            std::cout << "node->children is empty" << std::endl;
            return std::make_pair(-1, nullptr);
        }

        // Find the action with the maximum score
        float argmax = -std::numeric_limits<float>::infinity();
        int action = -1;
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            Node* child_tmp = kv.second;
            if (to_argmax[action_tmp] > argmax) {
                argmax = to_argmax[action_tmp];
                action = action_tmp;
            }
        }

        // Check if action has been updated
        if (action == -1) {
            // If no action is updated, return a random valid action and corresponding node
            std::cout << "action == -1, selecting a random valid action" << std::endl;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> distrib(0, node->children.size() - 1);

            auto it = std::next(std::begin(node->children), distrib(gen));
            return std::make_pair(it->first, it->second);
        }

        // Check if action is a valid key
        auto it = node->children.find(action);
        if (it != node->children.end()) {
            // If the action is valid, return it and the corresponding node
            return std::make_pair(action, it->second);
        } else {
            // If not a valid action, return default value or throw exception as needed
            std::cout << "action is not a valid key" << std::endl;
            return std::make_pair(-1, nullptr);
        }
    }


    // compute completed q value for each node
    std::vector<float> _qtransform_completed_by_mix_value(Node* node, std::vector<Node*> child_list) {
        // get q value and prior and visit count of children nodes
        std::vector<float> qvalues;
        std::vector<double> priors;
        std::vector<int> visit_counts;
        for (const auto& child : child_list) {
            qvalues.push_back(child->get_value());
            priors.push_back(child->prior_p);
            visit_counts.push_back(child->visit_count);
        }
        // softmax priors
        std::vector<double> priors_softmax = softmax(priors, 1);
        float value = _compute_mixed_value(node->raw_value, qvalues, visit_counts, priors);

        // calculate completed_value
        // the completed_value is qvalues where visit_counts > 0, otherwise it is value
        std::vector<float> completed_value;
        for (int i = 0; i < visit_counts.size(); i++) {
            if (visit_counts[i] > 0) {
                completed_value.push_back(qvalues[i]);
            } else {
                completed_value.push_back(value);
            }
        }

        // rescale the q value with max-min normalization
        std::vector<float> rescaled_qvalues = _rescale_qvalue(completed_value);

        // get max visit count
        int max_visit_count = *std::max_element(visit_counts.begin(), visit_counts.end());
        int visit_scale = maxvisit_init + max_visit_count;

        // scale the q value with visit_scale and value_scale
        std::vector<float> scaled_qvalues;
        for (int i = 0; i < rescaled_qvalues.size(); i++) {
            scaled_qvalues.push_back(rescaled_qvalues[i] * visit_scale * value_scale);
        }

        return scaled_qvalues;
    }

    //compute mixed value for each node
    float _compute_mixed_value(float raw_value, std::vector<float> qvalues, std::vector<int> visit_counts, std::vector<double> priors) {
        double min_num = -10e7;
        // calculate the sum of visit_counts
        int sum_visit_counts = std::accumulate(visit_counts.begin(), visit_counts.end(), 0);
        // ensure the child prior to be larger than min_num
        for (int i = 0; i < priors.size(); i++) {
            priors[i] = std::max(priors[i], min_num);
        }
        // calculate the sum of prior when the corresponding visit_count > 0
        double sum_prior = 0;
        for (int i = 0; i < priors.size(); i++) {
            if (visit_counts[i] > 0) {
                sum_prior += priors[i];
            }
        }
        // calculate weighted q sum
        float weighted_q_sum = 0;
        for (int i = 0; i < qvalues.size(); i++) {
            if (visit_counts[i] > 0){
                weighted_q_sum += qvalues[i] * priors[i] / sum_prior;
            }
        }

        return (raw_value + sum_visit_counts * weighted_q_sum) / (1 + sum_visit_counts);
    }

    // rescale the q value with max-min normalization
    std::vector<float> _rescale_qvalue(std::vector<float> qvalues) {
        float max_qvalue = *std::max_element(qvalues.begin(), qvalues.end());
        float min_qvalue = *std::min_element(qvalues.begin(), qvalues.end());
        std::vector<float> rescaled_qvalues;
        float epsilon = 1e-8;
        float gap = std::max(max_qvalue - min_qvalue, epsilon);
        for (int i = 0; i < qvalues.size(); i++) {
            rescaled_qvalues.push_back((qvalues[i] - min_qvalue) / gap);
        }
        return rescaled_qvalues;
    }

    std::vector<int> get_sequence_of_considered_visits(int max_num_considered_actions, int num_simulations)
    {
        /*
        Overview:
            Calculate the considered visit sequence.
        Arguments:
            - max_num_considered_actions: the maximum number of considered actions.
            - num_simulations: the upper limit number of simulations.
        Outputs:
            - the considered visit sequence.
        */
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

    std::vector<std::vector<int> > get_table_of_considered_visits(int max_num_considered_actions, int num_simulations)
    {
        /*
        Overview:
            Calculate the table of considered visits.
        Arguments:
            - max_num_considered_actions: the maximum number of considered actions.
            - num_simulations: the upper limit number of simulations.
        Outputs:
            - the table of considered visits.
        */
        std::vector<std::vector<int> > table;
        for (int m=0;m < max_num_considered_actions+1;m++){
            table.push_back(get_sequence_of_considered_visits(m, num_simulations));
        }
        return table;
    }

    //define _score_considered
    std::vector<float> _score_considered(int considered_visit, std::vector<Node*> child_list, std::vector<float> completed_qvalues) {
        /*
        Overview:
            Calculate the score of considered visits.
        Arguments:
            - considered_visit: the considered visit.
            - child_list: the list of child nodes.
            - completed_qvalues: the completed q values.
        Outputs:
            - the score of considered visits.
        */
        float low_logit = -1e9;
        // get max priors from child_list
        std::vector<float> priors;
        for (const auto& child : child_list) {
            priors.push_back(child->prior_p);
        }
        float max_prior = *std::max_element(priors.begin(), priors.end());
        // each priors minus max_logits
        for (int i = 0; i < priors.size(); i++) {
            priors[i] -= max_prior;
        }
        std::vector<float> penalty;
        // if the visit count of child node is equal to considered_visit, add 0 to penalty, else add -infinity
        for (int i = 0; i < child_list.size(); i++) {
            if (child_list[i]->visit_count == considered_visit) {
                penalty.push_back(0);
            } else {
                penalty.push_back(-std::numeric_limits<float>::infinity());
            }
        }

        // calculate the score of considered visits
        std::vector<float> score_considered;
        for (int i = 0; i < child_list.size(); i++) {
            score_considered.push_back(std::max(low_logit, gumbel[i] + priors[i] + penalty[i] + completed_qvalues[i]));
        }

        return score_considered;
    }
    
    // This function expands a leaf node by generating its children based on the legal actions and their prior probabilities.
    double _expand_leaf_node(Node* node, py::object simulate_env, py::object policy_forward_fn) {
         if (node == nullptr) {
        throw std::runtime_error("Encountered null node in _expand_leaf_node");
        }
        std::map<int, double> action_probs_dict;
        double leaf_value;
        py::tuple result = policy_forward_fn(simulate_env);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();
        // record raw value to node for gumbel alphazero
        node->raw_value = leaf_value;
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // only for debug
        // std::cout << "position18 " << std::endl;
        // auto legal_actions_pyobj = simulate_env.attr("legal_actions_func")();
        // std::cout << "position15 " << std::endl;
        // if (!py::isinstance<py::iterable>(legal_actions_pyobj)) {
        //     throw std::runtime_error("legal_actions did not return an iterable object");
        // }
        // py::list legal_actions_list = legal_actions_pyobj.cast<py::list>();
        // std::vector<int> legal_actions;
        // for (auto item : legal_actions_list) {
        //     if (py::isinstance<py::int_>(item)) {
        //         legal_actions.push_back(item.cast<int>());
        //     } else {
        //         throw std::runtime_error("Non-integer item in legal_actions list");
        //     }
        // }

        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                node->children[action] = new Node(node, prior_p);
            }
        }
        return leaf_value;
    }
    
    // This function returns the next action to take and the probabilities of each action based on the current state and the policy-value function.
    std::tuple<int, std::vector<double>, std::vector<double>> get_next_action(py::object state_config_for_env_reset, py::object policy_forward_fn, double temperature, bool sample) {
        Node* root = new Node();
        py::object init_state = state_config_for_env_reset["init_state"];
        if (!init_state.is_none()) {
            init_state = py::bytes(init_state.attr("tobytes")());
        }
        py::object katago_game_state = state_config_for_env_reset["katago_game_state"];
        if (!katago_game_state.is_none()) {
        // TODO(pu): polish efficiency
            katago_game_state = py::module::import("pickle").attr("dumps")(katago_game_state);
        }
        simulate_env.attr("reset")(
            state_config_for_env_reset["start_player_index"].cast<int>(),
            init_state,
            state_config_for_env_reset["katago_policy_init"].cast<bool>(),
            katago_game_state
        );

        _expand_leaf_node(root, simulate_env, policy_forward_fn);
        if (sample) {
            _add_exploration_noise(root);
        }

        for (int n = 0; n < num_simulations; ++n) {
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");
            _simulate(root, simulate_env, policy_forward_fn);
        }

        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (root->children.count(action)) {
                action_visits.push_back(std::make_pair(action, root->children[action]->visit_count));
            } else {
                action_visits.push_back(std::make_pair(action, 0));
            }
        }

        // Convert 'action_visits' into two separate arrays.
        std::vector<int> actions;
        std::vector<int> visits;
        for (const auto& av : action_visits) {
            actions.push_back(av.first);
            visits.push_back(av.second);
        }

        std::vector<double> visits_d(visits.begin(), visits.end());
        std::vector<double> action_probs = visit_count_to_action_distribution(visits_d, temperature);

        int action;
        if (sample) {
            action = random_choice(actions, action_probs);
        } else {
            action = actions[std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()))];
        }

        std::vector<double> improved_probs = _get_improved_policy(root);
        return std::tuple<int, std::vector<double>, std::vector<double>>(action, action_probs, improved_probs);

    }

    std::vector<double> _get_improved_policy(Node* root){
        double infymin = -std::numeric_limits<float>::infinity();
        // get visit count and prior of the child nodes
        std::vector<int> visit_counts;
        std::vector<double> priors;
        std::vector<Node*> children;
        std::vector<int> actions;
        for (const auto& kv : root->children) {
            visit_counts.push_back(kv.second->visit_count);
            priors.push_back((double)kv.second->prior_p);
            children.push_back(kv.second);
            actions.push_back(kv.first);
        }

        // get qtransform completed value
        std::vector<float> completed_value = _qtransform_completed_by_mix_value(root, children);

        // calculate probs
        std::vector<double> probs(simulate_env.attr("action_space").attr("n").cast<int>(), infymin);
        for (int i = 0; i < actions.size(); ++i) {
            int action = actions[i];
            probs[action] = priors[i] + completed_value[i];
        }

        // softmax probs
        std::vector<double> probs_softmax = softmax(probs, 1);
        return probs_softmax;
    }

    void _simulate(Node* node, py::object simulate_env, py::object policy_forward_fn) {
        while (!node->is_leaf()) {
            int action;
            if (node->is_root()){
                std::tie(action, node) = _select_root_child(node, simulate_env);
                if (node == nullptr) {
                    throw std::runtime_error("Encountered null node in _select_root_child");
                }
            }
            else{
                std::tie(action, node) = _select_interior_child(node, simulate_env);
                if (node == nullptr) {
                    throw std::runtime_error("Encountered null node in _select_interior_child");
                }
            }
            if (action == -1) {
                break;
            }
            simulate_env.attr("step")(action);
        }

        bool done;
        int winner;
        py::tuple result = simulate_env.attr("get_done_winner")();
        done = result[0].cast<bool>();
        winner = result[1].cast<int>();

        double leaf_value;
        if (!done) {
            if (node == nullptr) {
                throw std::runtime_error("Encountered null node in !done");
            }
            leaf_value = _expand_leaf_node(node, simulate_env, policy_forward_fn);
        }

        else {
             if (simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>() == "self_play_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                } else {
                    leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                }
            }
            else if (simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>() == "play_with_bot_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                } else if (winner == 1) {
                    leaf_value = 1;
                } else if (winner == 2) {
                    leaf_value = -1;
                }
            }
        }

    if (simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>() == "play_with_bot_mode") {
        node->update_recursive(leaf_value, simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>());
    } else if (simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>() == "self_play_mode") {
        node->update_recursive(-leaf_value, simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>());
    }
    }


private:
    static std::vector<double> visit_count_to_action_distribution(const std::vector<double>& visits, double temperature) {
        // Check if temperature is 0
        if (temperature == 0) {
            throw std::invalid_argument("Temperature cannot be 0");
        }

        // Check if all visit counts are 0
        if (std::all_of(visits.begin(), visits.end(), [](double v){ return v == 0; })) {
            throw std::invalid_argument("All visit counts cannot be 0");
        }

        std::vector<double> normalized_visits(visits.size());

        // Divide visit counts by temperature
        for (size_t i = 0; i < visits.size(); i++) {
            normalized_visits[i] = visits[i] / temperature;
        }

        // Calculate the sum of all normalized visit counts
        double sum = std::accumulate(normalized_visits.begin(), normalized_visits.end(), 0.0);

        // Normalize the visit counts
        for (double& visit : normalized_visits) {
            visit /= sum;
        }

        return normalized_visits;
    }

static std::vector<double> softmax(const std::vector<double>& values, double temperature) {
    if (values.empty()) {
        return {};
    }

    if (temperature == 0.0) {
        throw std::invalid_argument("Temperature cannot be zero");
    }

    std::vector<double> exps;
    exps.reserve(values.size());

    // Compute the maximum value
    double max_value = *std::max_element(values.begin(), values.end());

    // Subtract the maximum value and divide by temperature before exponentiating, for numerical stability
    for (double v : values) {
        double scaled_value = (v - max_value) / temperature;
        exps.push_back(std::exp(scaled_value));
    }

    // Use Kahan summation to compute the sum of exps
    double sum = 0.0;
    double c = 0.0;  // Running compensation for lost low-order bits
    for (double exp_v : exps) {
        double y = exp_v - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    for (double& exp_v : exps) {
        exp_v /= sum;
    }

    return exps;
}



    static int random_choice(const std::vector<int>& actions, const std::vector<double>& probs) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(probs.begin(), probs.end());
        return actions[d(gen)];
    }

};

// This function uses pybind11 to expose the Node and MCTS classes to Python.
// This allows Python code to create and manipulate instances of these classes.
PYBIND11_MODULE(mcts_gumbel_alphazero, m) {
    py::class_<Node>(m, "Node")
        .def(py::init([](Node* parent, float prior_p){
        return new Node(parent ? parent : nullptr, prior_p);
        }), py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def_property_readonly("value", &Node::get_value)
        .def("update", &Node::update)
        .def("update_recursive", &Node::update_recursive)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def("parent", &Node::get_parent)
        .def_readwrite("prior_p", &Node::prior_p)
        .def_readwrite("children", &Node::children)
        .def("add_child", &Node::add_child)
        .def_readwrite("visit_count", &Node::visit_count)
        .def_readwrite("raw_value", &Node::raw_value);



    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, int, double, double, double, double, int, float, float, float, int, py::object>(),
             py::arg("max_moves")=512, py::arg("num_simulations")=800,
             py::arg("pb_c_base")=19652, py::arg("pb_c_init")=1.25,
             py::arg("root_dirichlet_alpha")=0.3, py::arg("root_noise_weight")=0.25,
             py::arg("maxvisit_init")=50, py::arg("value_scale")=0.1,
             py::arg("gumbel_scale")=10.0, py::arg("gumbel_rng")=0.0,
             py::arg("max_num_considered_actions")=4,
             py::arg("simulate_env"))
        .def("_ucb_score", &MCTS::_ucb_score)
        .def("_add_exploration_noise", &MCTS::_add_exploration_noise)
        .def("_generate_gumbel", &MCTS::_generate_gumbel)
        .def("_select_child", &MCTS::_select_child)
        .def("_select_root_child", &MCTS::_select_root_child)
        .def("_select_interior_child", &MCTS::_select_interior_child)
        .def("_qtransform_completed_by_mix_value", &MCTS::_qtransform_completed_by_mix_value)
        .def("_compute_mixed_value", &MCTS::_compute_mixed_value)
        .def("_rescale_qvalue", &MCTS::_rescale_qvalue)
        .def("get_sequence_of_considered_visits", &MCTS::get_sequence_of_considered_visits)
        .def("get_table_of_considered_visits", &MCTS::get_table_of_considered_visits)
        .def("_score_considered", &MCTS::_score_considered)
        .def("_expand_leaf_node", &MCTS::_expand_leaf_node)
        .def("get_next_action", &MCTS::get_next_action)
        .def("_get_improved_policy", &MCTS::_get_improved_policy)
        .def("_simulate", &MCTS::_simulate);
}