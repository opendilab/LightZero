#include <map>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>

class Node {
public:
    // Constructor, initializes a Node with a parent pointer and a prior probability
    Node(Node* parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value_sum(0.0) {}

    // Destructor, deletes all child nodes when a node is deleted to prevent memory leaks
    ~Node() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }

    // Returns the average value of the node
    float get_value() {
        return visit_count == 0 ? 0.0 : value_sum / visit_count;
    }

    // Updates the visit count and value sum of the node
    void update(float value) {
        visit_count++;
        value_sum += value;
    }

    // Recursively updates the value and visit count of the node and its parent nodes
    void update_recursive(float leaf_value, std::string battle_mode_in_simulation_env) {
        // If the mode is "self_play_mode", the leaf_value is subtracted from the parent's value
        if (battle_mode_in_simulation_env == "self_play_mode") {
            update(leaf_value);
            if (!is_root()) {
                parent->update_recursive(-leaf_value, battle_mode_in_simulation_env);
            }
        }
        // If the mode is "play_with_bot_mode", the leaf_value is added to the parent's value
        else if (battle_mode_in_simulation_env == "play_with_bot_mode") {
            update(leaf_value);
            if (!is_root()) {
                parent->update_recursive(leaf_value, battle_mode_in_simulation_env);
            }
        }
    }

    // Returns true if the node has no children
    bool is_leaf() {
        return children.empty();
    }

    // Returns true if the node has no parent
    bool is_root() {
        return parent == nullptr;
    }

    // Returns a pointer to the node's parent
    Node* get_parent() {
        return parent;
    }

    // Returns a map of the node's children
    std::map<int, Node*> get_children() {
        return children;
    }

    // Returns the node's visit count
    int get_visit_count() {
        return visit_count;
    }

    // Adds a child to the node
    void add_child(int action, Node* node) {
        children[action] = node;
    }

public:
    Node* parent;  // Pointer to the parent node
    float prior_p;  // Prior probability of the node
    int visit_count;  // Count of visits to the node
    float value_sum;  // Sum of values of the node
    std::map<int, Node*> children;  // Map of child nodes
};