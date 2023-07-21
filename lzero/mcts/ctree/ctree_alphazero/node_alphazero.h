#include <map>
#include <string>

class Node {
public:
    Node(Node* parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value_sum(0.0) {}

    float get_value() {
        return visit_count == 0 ? 0.0 : value_sum / visit_count;
    }

    void update(float value) {
        visit_count++;
        value_sum += value;
    }

    void update_recursive(float leaf_value, std::string mcts_mode) {
        update(leaf_value);
        if (!is_root()) {
            if (mcts_mode == "self_play_mode") {
                parent->update_recursive(-leaf_value, mcts_mode);
            }
            else if (mcts_mode == "play_with_bot_mode") {
                parent->update_recursive(leaf_value, mcts_mode);
            }
        }
    }

    bool is_leaf() {
        return children.empty();
    }

    bool is_root() {
        return parent == nullptr;
    }

    Node* get_parent() {
        return parent;
    }

    std::map<int, Node*> get_children() {
        return children;
    }

    int get_visit_count() {
        return visit_count;
    }

public:
    Node* parent;
    float prior_p;
    int visit_count;
    float value_sum;
    std::map<int, Node*> children; // or std::vector<Node*>

// private:
//     Node* _parent;
//     float _prior_p;
//     int _visit_count;
//     float _value_sum;
//     std::map<int, Node*> _children;
};