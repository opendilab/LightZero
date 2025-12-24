/**
 * Overview:
 *     This module implements Monte Carlo Tree Search (MCTS) for AlphaZero-like algorithms
 *     with batch processing support for improved GPU utilization.
 *
 * Current Architecture:
 *     - MCTS tree search implemented in C++ for performance
 *     - Environment simulation via Python objects for modularity
 *     - Frequent C++-Python boundary crossings for env interactions
 *
 * Performance Bottleneck & Future Optimization:
 *     Frequent C++-Python interactions (env.step, legal_actions, get_done_winner, etc.)
 *     may become a bottleneck at high batch sizes. Future optimization direction:
 *     Consider C++-ifying the environment (env_cpp_ification) to eliminate boundary
 *     crossing overhead and enable efficient batch operations.
 */

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

namespace py = pybind11;

// The MCTS class implements Monte Carlo Tree Search (MCTS) for AlphaZero-like algorithms.
class MCTS {

private:
    int max_moves;                  // Maximum allowed moves in a game
    int num_simulations;            // Number of MCTS simulations
    double pb_c_base;               // Coefficient for UCB exploration term (base)
    double pb_c_init;               // Coefficient for UCB exploration term (initial value)
    double root_dirichlet_alpha;    // Alpha parameter for Dirichlet noise
    double root_noise_weight;       // Weight for exploration noise added to root node
    py::object simulate_env;        // Python object representing the simulation environment

public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25, py::object simulate_env=py::none())
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight),
          simulate_env(simulate_env) {}

    // Getter for the simulation environment (Python object)
    py::object get_simulate_env() const {
        return simulate_env;
    }

    // Setter for the simulation environment
    void set_simulate_env(py::object env) {
        simulate_env = env;
    }

    // Getter for pb_c_base
    double get_pb_c_base() const { return pb_c_base; }

    // Getter for pb_c_init
    double get_pb_c_init() const { return pb_c_init; }

    // Calculate the Upper Confidence Bound (UCB) score for child nodes
    double _ucb_score(std::shared_ptr<Node> parent, std::shared_ptr<Node> child) {
        // Calculate PB-C component of UCB
        double pb_c = std::log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(parent->visit_count) / (child->visit_count + 1);

        // Combine prior probability and value score
        double prior_score = pb_c * child->prior_p;
        double value_score = child->get_value();
        return prior_score + value_score;
    }

    // Add Dirichlet noise to the root node for exploration
    void _add_exploration_noise(std::shared_ptr<Node> node) {
        std::vector<int> actions;
        // Collect all child actions of the root node
        for (const auto& kv : node->children) {
            actions.push_back(kv.first);
        }

        // Generate Dirichlet noise
        std::default_random_engine generator;
        std::gamma_distribution<double> distribution(root_dirichlet_alpha, 1.0);

        std::vector<double> noise;
        double sum = 0;
        for (size_t i = 0; i < actions.size(); ++i) {
            double sample = distribution(generator);
            noise.push_back(sample);
            sum += sample;
        }

        // Normalize the noise
        for (size_t i = 0; i < noise.size(); ++i) {
            noise[i] /= sum;
        }

        // Mix noise with prior probabilities
        double frac = root_noise_weight;
        for (size_t i = 0; i < actions.size(); ++i) {
            node->children[actions[i]]->prior_p = node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
        }
    }

    // Select the best child node based on UCB score
    std::pair<int, std::shared_ptr<Node>> _select_child(std::shared_ptr<Node> node, py::object simulate_env) {
        int action = -1;
        std::shared_ptr<Node> child = nullptr;
        double best_score = -9999999;

        // Get legal actions: use cached version if available, otherwise query environment
        // (Optimization: cached legal_actions reduce C++-Python boundary calls)
        std::vector<int> legal_actions;
        if (node->has_legal_actions()) {
            // Use cached legal actions from the node expansion
            legal_actions = node->get_legal_actions();
        } else {
            // Fallback: query environment if not cached (shouldn't happen in normal flow)
            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }
        }

        // Iterate through all children
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> child_tmp = kv.second;

            // Check if the action is legal and calculate UCB score
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                double score = _ucb_score(node, child_tmp);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    child = child_tmp;
                }
            }
        }
        // If no valid child is found, return the current node
        if (child == nullptr) {
            child = node;
        }
        return std::make_pair(action, child);
    }

    // Expand a single leaf node for non-batch case.
    double _expand_leaf_node(std::shared_ptr<Node> node, py::object simulate_env, py::object policy_value_func) {
        std::map<int, double> action_probs_dict;
        double leaf_value;

        // Call the policy-value function to get action probabilities and leaf value
        py::tuple result = policy_value_func(simulate_env);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();

        // Get the legal actions from the simulation environment
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // Cache legal actions in the node for future use (optimization: avoid repeated env queries)
        node->set_legal_actions(legal_actions);

        // Add child nodes for legal actions
        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                node->children[action] = std::make_shared<Node>(node, prior_p);
            }
        }

        return leaf_value;
    }

    // Batch expand multiple leaf nodes using parallel batch inference.
    // Returns a list of leaf values for all nodes.
    std::vector<double> _batch_expand_leaf_nodes(
        const std::vector<std::shared_ptr<Node>>& leaf_nodes,
        const std::vector<py::object>& simulate_envs,
        py::object policy_value_func_batch
    ) {
        if (leaf_nodes.empty() || simulate_envs.empty()) {
            return std::vector<double>();
        }

        int batch_size = leaf_nodes.size();
        std::vector<double> leaf_values(batch_size);

        py::list env_list;
        for (int i = 0; i < batch_size; ++i) {
            env_list.append(simulate_envs[i]);
        }

        py::list batch_results = policy_value_func_batch(env_list).cast<py::list>();

        for (int i = 0; i < batch_size; ++i) {
            py::object env = simulate_envs[i];
            std::shared_ptr<Node> node = leaf_nodes[i];

            py::tuple result = batch_results[i].cast<py::tuple>();
            std::map<int, double> action_probs_dict = result[0].cast<std::map<int, double>>();
            double leaf_value = result[1].cast<double>();

            leaf_values[i] = leaf_value;

            py::list legal_actions_list = env.attr("legal_actions").cast<py::list>();
            std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

            // Cache legal actions in the node for future use (optimization: avoid repeated env queries)
            node->set_legal_actions(legal_actions);

            for (const auto& kv : action_probs_dict) {
                int action = kv.first;
                double prior_p = kv.second;
                if (std::find(legal_actions.begin(), legal_actions.end(), action) !=
                    legal_actions.end()) {
                    node->children[action] = std::make_shared<Node>(node, prior_p);
                }
            }
        }

        return leaf_values;
    }

    // Batch version: Get next actions for multiple environments with batch inference optimization.
    std::vector<std::tuple<int, std::vector<double>, std::shared_ptr<Node>>> get_next_actions_batch(
        py::list state_configs_list,
        py::object policy_value_func_batch,
        double temperature,
        bool sample,
        py::list simulate_env_list
    ) {
        int batch_size = py::len(state_configs_list);
        std::vector<std::tuple<int, std::vector<double>, std::shared_ptr<Node>>> results;
        results.reserve(batch_size);

        std::vector<std::shared_ptr<Node>> roots;
        roots.reserve(batch_size);

        std::vector<py::object> init_states;
        std::vector<py::object> katago_game_states;
        init_states.reserve(batch_size);
        katago_game_states.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            roots.push_back(std::make_shared<Node>());
            py::object state_config = state_configs_list[i].cast<py::object>();

            py::object init_state = state_config["init_state"];
            if (!init_state.is_none()) {
                init_state = py::bytes(init_state.attr("tobytes")());
            }
            init_states.push_back(init_state);

            py::object katago_game_state = state_config["katago_game_state"];
            if (!katago_game_state.is_none()) {
                katago_game_state = py::module::import("pickle").attr("dumps")(katago_game_state);
            }
            katago_game_states.push_back(katago_game_state);
        }

        py::list env_list;
        for (int i = 0; i < batch_size; ++i) {
            py::object state_config = state_configs_list[i].cast<py::object>();
            py::object env = simulate_env_list[i];
            env.attr("reset")(
                state_config["start_player_index"].cast<int>(),
                init_states[i],
                state_config["katago_policy_init"].cast<bool>(),
                katago_game_states[i]
            );
            env_list.append(env);
        }

        py::list batch_results = policy_value_func_batch(env_list).cast<py::list>();

        for (int i = 0; i < batch_size; ++i) {
            py::object env = simulate_env_list[i];

            py::tuple result = batch_results[i].cast<py::tuple>();
            std::map<int, double> action_probs_dict = result[0].cast<std::map<int, double>>();

            py::list legal_actions_list = env.attr("legal_actions").cast<py::list>();
            std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

            // Cache legal actions in the root node for future use (optimization: avoid repeated env queries)
            roots[i]->set_legal_actions(legal_actions);

            for (const auto& kv : action_probs_dict) {
                if (std::find(legal_actions.begin(), legal_actions.end(), kv.first) !=
                    legal_actions.end()) {
                    roots[i]->children[kv.first] = std::make_shared<Node>(roots[i], kv.second);
                }
            }

            if (sample) {
                _add_exploration_noise(roots[i]);
            }
        }

        for (int n = 0; n < num_simulations; ++n) {
            std::vector<SimulationResult> simulation_results;
            simulation_results.reserve(batch_size);

            for (int i = 0; i < batch_size; ++i) {
                py::object state_config = state_configs_list[i].cast<py::object>();
                py::object env = simulate_env_list[i];

                env.attr("reset")(
                    state_config["start_player_index"].cast<int>(),
                    init_states[i],
                    state_config["katago_policy_init"].cast<bool>(),
                    katago_game_states[i]
                );
                env.attr("battle_mode") = env.attr("battle_mode_in_simulation_env");

                SimulationResult sim_result = _simulate_to_leaf(roots[i], env);
                simulation_results.push_back(sim_result);
            }

            std::vector<int> unfinished_indices;
            std::vector<std::shared_ptr<Node>> leaf_nodes_to_expand;
            std::vector<py::object> envs_to_infer;

            for (int i = 0; i < batch_size; ++i) {
                if (!simulation_results[i].is_done) {
                    unfinished_indices.push_back(i);
                    leaf_nodes_to_expand.push_back(simulation_results[i].leaf_node);
                    envs_to_infer.push_back(simulation_results[i].simulate_env);
                }
            }

            std::vector<double> leaf_values;
            if (!unfinished_indices.empty()) {
                leaf_values = _batch_expand_leaf_nodes(
                    leaf_nodes_to_expand,
                    envs_to_infer,
                    policy_value_func_batch
                );
            }

            for (int i = 0; i < batch_size; ++i) {
                std::shared_ptr<Node> leaf_node = simulation_results[i].leaf_node;
                py::object env = simulation_results[i].simulate_env;
                double leaf_value;

                if (simulation_results[i].is_done) {
                    std::string battle_mode =
                        env.attr("battle_mode_in_simulation_env").cast<std::string>();
                    int winner = simulation_results[i].winner;

                    if (battle_mode == "self_play_mode") {
                        if (winner == -1) {
                            leaf_value = 0;
                        } else {
                            leaf_value =
                                (env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                        }
                    }
                    else if (battle_mode == "play_with_bot_mode") {
                        if (winner == -1) {
                            leaf_value = 0;
                        }
                        else if (winner == 1) {
                            leaf_value = 1;
                        }
                        else if (winner == 2) {
                            leaf_value = -1;
                        }
                    }
                } else {
                    auto it = std::find(unfinished_indices.begin(), unfinished_indices.end(), i);
                    if (it != unfinished_indices.end()) {
                        int result_idx = std::distance(unfinished_indices.begin(), it);
                        leaf_value = leaf_values[result_idx];
                    } else {
                        leaf_value = 0;
                    }
                }

                std::string battle_mode =
                    env.attr("battle_mode_in_simulation_env").cast<std::string>();
                if (battle_mode == "play_with_bot_mode") {
                    leaf_node->update_recursive(leaf_value, battle_mode);
                }
                else if (battle_mode == "self_play_mode") {
                    leaf_node->update_recursive(-leaf_value, battle_mode);
                }
            }
        }

        for (int i = 0; i < batch_size; ++i) {
            py::object state_config = state_configs_list[i].cast<py::object>();
            py::object env = simulate_env_list[i];

            env.attr("reset")(
                state_config["start_player_index"].cast<int>(),
                init_states[i],
                state_config["katago_policy_init"].cast<bool>(),
                katago_game_states[i]
            );

            std::vector<std::pair<int, int>> action_visits;
            int action_space_n = env.attr("action_space").attr("n").cast<int>();
            for (int action = 0; action < action_space_n; ++action) {
                if (roots[i]->children.count(action)) {
                    action_visits.emplace_back(action, roots[i]->children[action]->visit_count);
                } else {
                    action_visits.emplace_back(action, 0);
                }
            }

            std::vector<int> actions;
            std::vector<int> visits;
            for (const auto& av : action_visits) {
                actions.emplace_back(av.first);
                visits.emplace_back(av.second);
            }

            std::vector<double> visits_d(visits.begin(), visits.end());
            std::vector<double> action_probs =
                visit_count_to_action_distribution(visits_d, temperature);

            int action_selected;
            if (sample) {
                action_selected = random_choice(actions, action_probs);
            } else {
                auto max_it = std::max_element(action_probs.begin(), action_probs.end());
                action_selected = actions[std::distance(action_probs.begin(), max_it)];
            }

            results.push_back(std::make_tuple(action_selected, action_probs, roots[i]));
        }

        return results;
    }

    // Non-batch version: Get the next action from MCTS for a single environment
    // For batch processing, use get_next_actions_batch() instead
    std::tuple<int, std::vector<double>, std::shared_ptr<Node>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
        std::shared_ptr<Node> root = std::make_shared<Node>();

        // Configure initial environment state
        py::object init_state = state_config_for_env_reset["init_state"];
        if (!init_state.is_none()) {
            init_state = py::bytes(init_state.attr("tobytes")());
        }
        py::object katago_game_state = state_config_for_env_reset["katago_game_state"];
        if (!katago_game_state.is_none()) {
            katago_game_state = py::module::import("pickle").attr("dumps")(katago_game_state);
        }
        simulate_env.attr("reset")(
            state_config_for_env_reset["start_player_index"].cast<int>(),
            init_state,
            state_config_for_env_reset["katago_policy_init"].cast<bool>(),
            katago_game_state
        );

        // Expand the root node
        _expand_leaf_node(root, simulate_env, policy_value_func);
        if (sample) {
            _add_exploration_noise(root);
        }

        // Run MCTS simulations
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

        // Collect visit counts from the root's children
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (root->children.count(action)) {
                action_visits.emplace_back(action, root->children[action]->visit_count);
            }
            else {
                action_visits.emplace_back(action, 0);
            }
        }

        std::vector<int> actions;
        std::vector<int> visits;
        for (const auto& av : action_visits) {
            actions.emplace_back(av.first);
            visits.emplace_back(av.second);
        }

        std::vector<double> visits_d(visits.begin(), visits.end());
        std::vector<double> action_probs = visit_count_to_action_distribution(visits_d, temperature);

        int action_selected;
        if (sample) {
            action_selected = random_choice(actions, action_probs);
        }
        else {
            action_selected = actions[std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()))];
        }

        // Return the selected action, action probabilities, and root node
        return std::make_tuple(action_selected, action_probs, root);
    }

    // Structure to store single simulation result for parallel batch inference.
    struct SimulationResult {
        std::shared_ptr<Node> leaf_node;
        bool is_done;
        int winner;
        py::object simulate_env;
    };

    // Single simulation from root to leaf node without inference, returns leaf node info.
    SimulationResult _simulate_to_leaf(std::shared_ptr<Node> node, py::object simulate_env) {
        while (!node->is_leaf()) {
            int action;
            std::shared_ptr<Node> child;
            std::tie(action, child) = _select_child(node, simulate_env);
            if (action == -1) {
                break;
            }
            simulate_env.attr("step")(action);
            node = child;
        }

        bool done;
        int winner;
        py::tuple result = simulate_env.attr("get_done_winner")();
        done = result[0].cast<bool>();
        winner = result[1].cast<int>();

        return SimulationResult{node, done, winner, simulate_env};
    }

    // Legacy single environment simulation function (kept for backward compatibility).
    void _simulate(std::shared_ptr<Node> node, py::object simulate_env, py::object policy_value_func) {
        while (!node->is_leaf()) {
            int action;
            std::shared_ptr<Node> child;
            std::tie(action, child) = _select_child(node, simulate_env);
            if (action == -1) {
                break;
            }
            simulate_env.attr("step")(action);
            node = child;
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
            std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
            if (battle_mode == "self_play_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                } else {
                    leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                }
            }
            else if (battle_mode == "play_with_bot_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                }
                else if (winner == 1) {
                    leaf_value = 1;
                }
                else if (winner == 2) {
                    leaf_value = -1;
                }
            }
        }

        std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
        if (battle_mode == "play_with_bot_mode") {
            node->update_recursive(leaf_value, battle_mode);
        }
        else if (battle_mode == "self_play_mode") {
            node->update_recursive(-leaf_value, battle_mode);
        }
    }

private:
    // Helper: Convert visit counts to action probabilities using temperature
    static std::vector<double> visit_count_to_action_distribution(const std::vector<double>& visits, double temperature) {
        if (temperature == 0) {
            throw std::invalid_argument("Temperature cannot be 0");
        }

        if (std::all_of(visits.begin(), visits.end(), [](double v){ return v == 0; })) {
            throw std::invalid_argument("All visit counts cannot be 0");
        }

        std::vector<double> normalized_visits(visits.size());

        for (size_t i = 0; i < visits.size(); i++) {
            normalized_visits[i] = visits[i] / temperature;
        }

        double sum = std::accumulate(normalized_visits.begin(), normalized_visits.end(), 0.0);

        for (double& visit : normalized_visits) {
            visit /= sum;
        }

        return normalized_visits;
    }

    // Helper: Softmax function to normalize values
    static std::vector<double> softmax(const std::vector<double>& values, double temperature) {
        std::vector<double> exps;
        double sum = 0.0;
        double max_value = *std::max_element(values.begin(), values.end());

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

    // Helper: Randomly choose an action based on probabilities
    static int random_choice(const std::vector<int>& actions, const std::vector<double>& probs) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(probs.begin(), probs.end());
        return actions[d(gen)];
    }
};

// Bind Node and MCTS to the same pybind11 module
PYBIND11_MODULE(mcts_alphazero, m) {
    // Bind the Node class
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<std::shared_ptr<Node>, float>(),
             py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def_property_readonly("value", &Node::get_value)
        .def("update", &Node::update)
        .def("update_recursive", &Node::update_recursive)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def_property_readonly("parent", &Node::get_parent)
        .def_property_readonly("children", &Node::get_children)
        .def("add_child", &Node::add_child)
        .def_property_readonly("visit_count", &Node::get_visit_count)
        .def_readwrite("prior_p", &Node::prior_p);

    // Bind the MCTS class
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
        .def("_simulate", &MCTS::_simulate)
        .def_property("simulate_env", &MCTS::get_simulate_env, &MCTS::set_simulate_env)
        .def_property_readonly("pb_c_base", &MCTS::get_pb_c_base)
        .def_property_readonly("pb_c_init", &MCTS::get_pb_c_init)
        .def("get_next_action", &MCTS::get_next_action,
             py::arg("state_config_for_env_reset"),
             py::arg("policy_value_func"),
             py::arg("temperature"),
             py::arg("sample"))
        .def("get_next_actions_batch", &MCTS::get_next_actions_batch,
             py::arg("state_configs_list"),
             py::arg("policy_value_func_batch"),
             py::arg("temperature"),
             py::arg("sample"),
             py::arg("simulate_env_list"));
}
