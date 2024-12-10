// node_alphazero.h

#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>

class Node : public std::enable_shared_from_this<Node> {
public:
    // 父节点和子节点使用 shared_ptr 管理
    std::shared_ptr<Node> parent;
    std::map<int, std::shared_ptr<Node>> children;

    // 构造函数
    Node(std::shared_ptr<Node> parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value_sum(0.0) {}

    // 默认析构函数
    ~Node() = default;

    // 获取节点平均值
    float get_value() const {
        return visit_count == 0 ? 0.0 : value_sum / visit_count;
    }

    // 更新节点的访问计数和价值总和
    void update(float value) {
        visit_count++;
        value_sum += value;
    }

    // 递归更新节点和父节点的值
    void update_recursive(float leaf_value, const std::string& battle_mode_in_simulation_env) {
        if (battle_mode_in_simulation_env == "self_play_mode") {
            update(leaf_value);
            if (!is_root() && parent) {
                parent->update_recursive(-leaf_value, battle_mode_in_simulation_env);
            }
        }
        else if (battle_mode_in_simulation_env == "play_with_bot_mode") {
            update(leaf_value);
            if (!is_root() && parent) {
                parent->update_recursive(leaf_value, battle_mode_in_simulation_env);
            }
        }
    }

    // 判断是否为叶子节点
    bool is_leaf() const {
        return children.empty();
    }

    // 判断是否为根节点
    bool is_root() const {
        return parent == nullptr;
    }

    // 添加子节点
    void add_child(int action, std::shared_ptr<Node> node) {
        children[action] = node;
    }

    // 获取访问计数
    int get_visit_count() const { return visit_count; }

    // 获取父节点
    std::shared_ptr<Node> get_parent() const {
        return parent;
    }

    // 获取子节点
    const std::map<int, std::shared_ptr<Node>>& get_children() const {
        return children;
    }

public:
    float prior_p;        // 节点的 prior probability
    int visit_count;      // 访问计数
    float value_sum;      // 价值总和
};

#endif // NODE_ALPHAZERO_H