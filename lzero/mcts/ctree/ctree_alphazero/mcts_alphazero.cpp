// This code is a Python extension implemented in C++ using the pybind11 library.
// It's a Monte Carlo Tree Search (MCTS) algorithm with modifications based on Google's AlphaZero paper.
// MCTS is an algorithm for making optimal decisions in a certain class of combinatorial problems.
// It's most famously used in board games like chess, Go, and shogi.

// The following lines include the necessary headers to facilitate the implementation of the MCTS algorithm.
#include "node_alphazero.h"
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
    py::object simulate_env;

// This part defines the constructor of the MCTS class.
// The constructor initializes the member variables with the provided arguments or with their default values.
public:
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25, py::object simulate_env=py::none())
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight),
          simulate_env(simulate_env) {}

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

    // This function expands a leaf node by generating its children based on the legal actions and their prior probabilities.
    double _expand_leaf_node(Node* node, py::object simulate_env, py::object policy_value_func) {

        std::map<int, double> action_probs_dict;
        double leaf_value;
        py::tuple result = policy_value_func(simulate_env);

        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();


        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();


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
    std::pair<int, std::vector<double>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
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

        _expand_leaf_node(root, simulate_env, policy_value_func);
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
            _simulate(root, simulate_env, policy_value_func);
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


        return std::make_pair(action, action_probs);
    }

    // This function performs a simulation from a given node until a leaf node is reached or a terminal state is reached.
    void _simulate(Node* node, py::object simulate_env, py::object policy_value_func) {
        while (!node->is_leaf()) {
            int action;
            std::tie(action, node) = _select_child(node, simulate_env);
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
            leaf_value = _expand_leaf_node(node, simulate_env, policy_value_func);
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
    }
    else if (simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>() == "self_play_mode") {
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
        std::vector<double> exps;
        double sum = 0.0;
        // Compute the maximum value
        double max_value = *std::max_element(values.begin(), values.end());

        // Subtract the maximum value before exponentiation, for numerical stability
        for (double v : values) {
            double exp_v = std::exp((v - max_value) / temperature);
            exps.push_back(exp_v);
            sum += exp_v;
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
PYBIND11_MODULE(mcts_alphazero, m) {
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
        .def_readwrite("visit_count", &Node::visit_count);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, int, double, double, double, double, py::object>(),
             py::arg("max_moves")=512, py::arg("num_simulations")=800,
             py::arg("pb_c_base")=19652, py::arg("pb_c_init")=1.25,
             py::arg("root_dirichlet_alpha")=0.3, py::arg("root_noise_weight")=0.25, py::arg("simulate_env"))
        .def("_ucb_score", &MCTS::_ucb_score)
        .def("_add_exploration_noise", &MCTS::_add_exploration_noise)
        .def("_select_child", &MCTS::_select_child)
        .def("_expand_leaf_node", &MCTS::_expand_leaf_node)
        .def("get_next_action", &MCTS::get_next_action)
        .def("_simulate", &MCTS::_simulate);
}