#ifndef MCTS_ALPHAZERO_BATCH_H
#define MCTS_ALPHAZERO_BATCH_H

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

// Batch版本的Roots类,管理多个root节点
class Roots {
public:
    std::vector<std::shared_ptr<Node>> roots;
    int num;  // batch size
    std::vector<std::vector<int>> legal_actions_list;

    Roots(int root_num, const std::vector<std::vector<int>>& legal_actions)
        : num(root_num), legal_actions_list(legal_actions) {
        for (int i = 0; i < root_num; ++i) {
            roots.push_back(std::make_shared<Node>());
        }
    }

    // 准备roots: 展开root节点,添加噪声
    void prepare(double root_noise_weight,
                 const std::vector<std::vector<double>>& noises,
                 const std::vector<double>& values,
                 const std::vector<std::vector<double>>& policy_logits_pool) {
        for (int i = 0; i < num; ++i) {
            auto& root = roots[i];
            const auto& legal_actions = legal_actions_list[i];
            const auto& policy_logits = policy_logits_pool[i];
            const auto& noise = noises[i];

            // 展开root节点 - 为每个合法动作创建子节点
            for (size_t j = 0; j < legal_actions.size(); ++j) {
                int action = legal_actions[j];
                double prior_p = policy_logits[action];

                // 应用dirichlet noise
                if (j < noise.size()) {
                    prior_p = prior_p * (1 - root_noise_weight) + noise[j] * root_noise_weight;
                }

                auto child = std::make_shared<Node>(root, prior_p);
                root->children[action] = child;
            }
        }
    }

    // 准备roots: 不添加噪声版本(用于evaluation)
    void prepare_no_noise(const std::vector<double>& values,
                          const std::vector<std::vector<double>>& policy_logits_pool) {
        for (int i = 0; i < num; ++i) {
            auto& root = roots[i];
            const auto& legal_actions = legal_actions_list[i];
            const auto& policy_logits = policy_logits_pool[i];

            // 展开root节点
            for (int action : legal_actions) {
                double prior_p = policy_logits[action];
                auto child = std::make_shared<Node>(root, prior_p);
                root->children[action] = child;
            }
        }
    }

    // 获取访问计数分布
    std::vector<std::vector<double>> get_distributions() const {
        std::vector<std::vector<double>> distributions;
        for (const auto& root : roots) {
            // 计算每个root的访问计数分布
            std::vector<double> dist;
            int total_visits = 0;

            // 先找最大的action索引
            int max_action = 0;
            for (const auto& kv : root->children) {
                max_action = std::max(max_action, kv.first);
            }

            dist.resize(max_action + 1, 0.0);

            for (const auto& kv : root->children) {
                int action = kv.first;
                auto child = kv.second;
                dist[action] = child->visit_count;
                total_visits += child->visit_count;
            }

            // 归一化
            if (total_visits > 0) {
                for (auto& v : dist) {
                    v /= total_visits;
                }
            }

            distributions.push_back(dist);
        }
        return distributions;
    }

    // 获取values
    std::vector<double> get_values() const {
        std::vector<double> values;
        for (const auto& root : roots) {
            values.push_back(root->get_value());
        }
        return values;
    }
};

// 结果包装器,用于在C++和Python之间传递数据
struct SearchResults {
    std::vector<int> latent_state_index_in_search_path;  // 叶节点在搜索路径中的深度
    std::vector<int> latent_state_index_in_batch;        // 叶节点对应的batch索引
    std::vector<int> last_actions;                        // 到达叶节点的动作
    std::vector<std::shared_ptr<Node>> leaf_nodes;       // 叶节点指针

    SearchResults(int batch_size) {
        latent_state_index_in_search_path.reserve(batch_size);
        latent_state_index_in_batch.reserve(batch_size);
        last_actions.reserve(batch_size);
        leaf_nodes.reserve(batch_size);
    }
};

// 计算UCB分数
double ucb_score(std::shared_ptr<Node> parent, std::shared_ptr<Node> child,
                 double pb_c_base, double pb_c_init) {
    double pb_c = std::log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= std::sqrt(parent->visit_count) / (child->visit_count + 1);

    double prior_score = pb_c * child->prior_p;
    double value_score = child->get_value();
    return prior_score + value_score;
}

// 选择子节点
std::pair<int, std::shared_ptr<Node>> select_child(
    std::shared_ptr<Node> node,
    const std::vector<int>& legal_actions,
    double pb_c_base, double pb_c_init) {

    int best_action = -1;
    std::shared_ptr<Node> best_child = nullptr;
    double best_score = -999999.0;

    for (int action : legal_actions) {
        if (node->children.count(action) == 0) {
            continue;
        }

        auto child = node->children[action];
        double score = ucb_score(node, child, pb_c_base, pb_c_init);

        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }

    return std::make_pair(best_action, best_child);
}

// 批量traverse: 对所有roots同时进行traverse,找到叶节点
SearchResults batch_traverse(
    Roots& roots,
    double pb_c_base,
    double pb_c_init,
    const std::vector<std::vector<int>>& current_legal_actions) {

    SearchResults results(roots.num);

    // 对每个环境的root进行traverse
    for (int batch_idx = 0; batch_idx < roots.num; ++batch_idx) {
        auto node = roots.roots[batch_idx];
        int depth = 0;
        int last_action = -1;

        std::vector<int> legal_actions = current_legal_actions[batch_idx];

        // 从root走到leaf
        while (!node->is_leaf() && depth < 100) {  // 添加最大深度防止无限循环
            int action;
            std::shared_ptr<Node> child;
            std::tie(action, child) = select_child(node, legal_actions, pb_c_base, pb_c_init);

            if (child == nullptr) {
                // 如果没有找到合法的子节点,停止
                break;
            }

            last_action = action;
            node = child;
            depth++;
        }

        // 记录结果
        results.latent_state_index_in_search_path.push_back(depth);
        results.latent_state_index_in_batch.push_back(batch_idx);
        results.last_actions.push_back(last_action);
        results.leaf_nodes.push_back(node);
    }

    return results;
}

// 批量backpropagate: 展开叶节点并反向传播
void batch_backpropagate(
    SearchResults& results,
    const std::vector<double>& values,
    const std::vector<std::vector<double>>& policy_logits_batch,
    const std::vector<std::vector<int>>& legal_actions_batch,
    const std::string& battle_mode) {

    for (size_t i = 0; i < results.leaf_nodes.size(); ++i) {
        auto leaf_node = results.leaf_nodes[i];
        double value = values[i];
        const auto& policy_logits = policy_logits_batch[i];
        const auto& legal_actions = legal_actions_batch[i];

        // 展开叶节点
        if (leaf_node->is_leaf()) {
            for (int action : legal_actions) {
                double prior_p = 0.0;
                if (action < static_cast<int>(policy_logits.size())) {
                    prior_p = policy_logits[action];
                }
                auto child = std::make_shared<Node>(leaf_node, prior_p);
                leaf_node->children[action] = child;
            }
        }

        // 反向传播
        leaf_node->update_recursive(value, battle_mode);
    }
}

// Python绑定
PYBIND11_MODULE(mcts_alphazero_batch, m) {
    m.doc() = "Batch MCTS implementation for AlphaZero";

    // 绑定Roots类
    py::class_<Roots>(m, "Roots")
        .def(py::init<int, const std::vector<std::vector<int>>&>())
        .def("prepare", &Roots::prepare,
             py::arg("root_noise_weight"),
             py::arg("noises"),
             py::arg("values"),
             py::arg("policy_logits_pool"))
        .def("prepare_no_noise", &Roots::prepare_no_noise,
             py::arg("values"),
             py::arg("policy_logits_pool"))
        .def("get_distributions", &Roots::get_distributions)
        .def("get_values", &Roots::get_values)
        .def_readonly("num", &Roots::num);

    // 绑定SearchResults类
    py::class_<SearchResults>(m, "SearchResults")
        .def(py::init<int>())
        .def_readonly("latent_state_index_in_search_path", &SearchResults::latent_state_index_in_search_path)
        .def_readonly("latent_state_index_in_batch", &SearchResults::latent_state_index_in_batch)
        .def_readonly("last_actions", &SearchResults::last_actions);

    // 绑定函数
    m.def("batch_traverse", &batch_traverse,
          py::arg("roots"),
          py::arg("pb_c_base"),
          py::arg("pb_c_init"),
          py::arg("current_legal_actions"),
          "Batch traverse multiple MCTS trees in parallel");

    m.def("batch_backpropagate", &batch_backpropagate,
          py::arg("results"),
          py::arg("values"),
          py::arg("policy_logits_batch"),
          py::arg("legal_actions_batch"),
          py::arg("battle_mode"),
          "Batch backpropagate values through multiple MCTS trees");
}

#endif // MCTS_ALPHAZERO_BATCH_H
