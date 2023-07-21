#include "node_alphazero.h"
#include <cmath>
#include <map>
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <iostream>

namespace py = pybind11;


class MCTS {
    int max_moves;
    int num_simulations;
    double pb_c_base;
    double pb_c_init;
    double root_dirichlet_alpha;
    double root_noise_weight;

public:
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25, 
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25)
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight) {}

    // 包括get_next_action，_simulate，_select_child，_expand_leaf_node，_ucb_score，_add_exploration_noise
    
    double _ucb_score(Node* parent, Node* child) {
        double pb_c = std::log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(parent->visit_count) / (child->visit_count + 1);

        double prior_score = pb_c * child->prior_p;
        // double value_score = child->value;
        double value_score = child->get_value();  // 使用get_value()方法替代value成员
        return prior_score + value_score;
    }

    void _add_exploration_noise(Node* node) {
        std::vector<int> actions;
        for (const auto& kv : node->children) {
            actions.push_back(kv.first);
        }

        std::default_random_engine generator;
        std::gamma_distribution<double> distribution(root_dirichlet_alpha, 1.0);
        
        std::vector<double> noise;
        for (size_t i = 0; i < actions.size(); ++i) {
            noise.push_back(distribution(generator));
        }
        
        double frac = root_noise_weight;
        for (size_t i = 0; i < actions.size(); ++i) {
            node->children[actions[i]]->prior_p = node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
        }
    }

    // 在MCTS类中定义_select_child和_expand_leaf_node函数
    std::pair<int, Node*> _select_child(Node* node, py::object simulate_env) {
    
        int action = -1;
        Node* child = nullptr;
        double best_score = -9999999;
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            Node* child_tmp = kv.second;
  
            // py::list legal_actions_py = simulate_env.attr("legal_actions")().cast<py::list>();
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

    double _expand_leaf_node(Node* node, py::object simulate_env, py::object policy_forward_fn) {
        // std::cout << "position11 " << std::endl;
    
        std::map<int, double> action_probs_dict;
        double leaf_value;
        // std::cout << "position12 " << std::endl;
        py::tuple result = policy_forward_fn(simulate_env);
        // std::cout << "position13 " << std::endl;

        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();

        // 获取 legal_actions 属性的值，并将其转换为一个 std::vector<int>
        // std::cout << "position14 " << std::endl;

        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // only for debug
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

        // std::cout << "position16 " << std::endl;

        for (const auto& kv : action_probs_dict) {
            // std::cout << "position17 " << std::endl;

            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                node->children[action] = new Node(node, prior_p);
            }
        }
        // std::cout << "position18 " << std::endl;

        return leaf_value;
    }

    std::pair<int, std::vector<double>> get_next_action(py::object simulate_env, py::object policy_forward_fn, double temperature, bool sample) {
        // std::cout << "position1 " << std::endl;
        Node* root = new Node();
        // std::cout << "position2 " << std::endl;
        _expand_leaf_node(root, simulate_env, policy_forward_fn);
        // std::cout << "position3 " << std::endl;

        if (sample) {
            // std::cout << "position4 " << std::endl;
            _add_exploration_noise(root);
            // std::cout << "position5 " << std::endl;

        }
        // std::cout << "position6 " << std::endl;

        for (int n = 0; n < num_simulations; ++n) {
            // std::cout << "position7 " << std::endl;
            py::object simulate_env_copy = simulate_env.attr("clone")();
            // std::cout << "position8 " << std::endl;
            _simulate(root, simulate_env_copy, policy_forward_fn);
            // std::cout << "position9 " << std::endl;
            simulate_env_copy = py::none();
            // std::cout << "position10 " << std::endl;
        }

        std::vector<std::pair<int, int>> action_visits;
            // std::cout << "position11 " << std::endl;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (root->children.count(action)) {
                // std::cout << "position12 " << std::endl;

                action_visits.push_back(std::make_pair(action, root->children[action]->visit_count));
            } else {
                // std::cout << "position13 " << std::endl;

                action_visits.push_back(std::make_pair(action, 0));
            }
        }

        // 转换action_visits为两个分离的数组
        std::vector<int> actions;
        std::vector<int> visits;
                // std::cout << "position14 " << std::endl;

        for (const auto& av : action_visits) {
            actions.push_back(av.first);
            visits.push_back(av.second);
        }

        // 计算action_probs
        std::vector<double> visit_logs;
        for (int v : visits) {
            visit_logs.push_back(std::log(v + 1e-10));
        }
        std::vector<double> action_probs = softmax(visit_logs, temperature);
        // std::cout << "position15 " << std::endl;

        // 根据action_probs选择一个action
        int action;
        if (sample) {
            action = random_choice(actions, action_probs);
        } else {
            action = actions[std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()))];
        }
        // std::cout << "position16 " << std::endl;

        return std::make_pair(action, action_probs);
    }

    void _simulate(Node* node, py::object simulate_env, py::object policy_forward_fn) {
        // std::cout << "position21 " << std::endl;
    
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
        // std::tie(done, winner) = simulate_env.attr("get_done_winner")();
        py::tuple result = simulate_env.attr("get_done_winner")();
        done = result[0].cast<bool>();
        winner = result[1].cast<int>();

        double leaf_value;
        // std::cout << "position22 " << std::endl;

        if (!done) {
            leaf_value = _expand_leaf_node(node, simulate_env, policy_forward_fn);
            // std::cout << "position23 " << std::endl;
        } 
        else {
            // if (simulate_env.attr("mcts_mode") == "self_play_mode") {
             if (simulate_env.attr("mcts_mode").cast<std::string>() == "self_play_mode") {  // 使用get_mcts_mode()方法替代mcts_mode成员
                // std::cout << "position24 " << std::endl;
                
                if (winner == -1) {
                    leaf_value = 0;
                } else {
                    leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                }
            } 
            else if (simulate_env.attr("mcts_mode").cast<std::string>() == "play_with_bot_mode") {  // 使用get_mcts_mode()方法替代mcts_mode成员
                if (winner == -1) {
                    leaf_value = 0;
                } else if (winner == 1) {
                    leaf_value = 1;
                } else if (winner == 2) {
                    leaf_value = -1;
                }
            }
        }
    // std::cout << "position25 " << std::endl;

    // if (simulate_env.attr("mcts_mode") == "play_with_bot_mode") {
    if (simulate_env.attr("mcts_mode").cast<std::string>() == "play_with_bot_mode") {
        node->update_recursive(leaf_value, simulate_env.attr("mcts_mode").cast<std::string>());
    } else if (simulate_env.attr("mcts_mode").cast<std::string>() == "self_play_mode") {
        node->update_recursive(-leaf_value, simulate_env.attr("mcts_mode").cast<std::string>());
    }
    }

private:
    static std::vector<double> softmax(const std::vector<double>& values, double temperature) {
        std::vector<double> exps;
        double sum = 0.0;
        for (double v : values) {
            double exp_v = std::exp(v / temperature);
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

PYBIND11_MODULE(mcts_alphazero, m) {
    py::class_<Node>(m, "Node")
        // .def(py::init<Node*, float>())
        .def(py::init([](Node* parent, float prior_p){
        return new Node(parent ? parent : nullptr, prior_p);
        }), py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def("value", &Node::get_value)
        .def("update", &Node::update)
        .def("update_recursive", &Node::update_recursive)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def("parent", &Node::get_parent)
        // .def("children", &Node::get_children)
        .def_readwrite("children", &Node::children)
        .def("visit_count", &Node::get_visit_count);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, int, double, double, double, double>(),
             py::arg("max_moves")=512, py::arg("num_simulations")=800,
             py::arg("pb_c_base")=19652, py::arg("pb_c_init")=1.25,
             py::arg("root_dirichlet_alpha")=0.3, py::arg("root_noise_weight")=0.25)
        .def("_ucb_score", &MCTS::_ucb_score)
        .def("_add_exploration_noise", &MCTS::_add_exploration_noise)
        .def("_select_child", &MCTS::_select_child)
        .def("_expand_leaf_node", &MCTS::_expand_leaf_node)
        // .def("get_next_action", &MCTS::get_next_action,
        //      py::arg("simulate_env"), py::arg("policy_forward_fn"),
        //      py::arg("temperature")=1.0, py::arg("sample")=true)
        .def("get_next_action", &MCTS::get_next_action)
        .def("_simulate", &MCTS::_simulate);
}


// 构建与编译命令
// mkdir build
// cd buld
// cmake ..
// make