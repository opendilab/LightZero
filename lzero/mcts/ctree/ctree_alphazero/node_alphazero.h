#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>
#include <iostream>

class Node : public std::enable_shared_from_this<Node> {
public:
    // Parent and child nodes are managed using shared_ptr
    std::shared_ptr<Node> parent;
    std::map<int, std::shared_ptr<Node>> children;

    // Constructor
    Node(std::shared_ptr<Node> parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value_sum(0.0) {}

    // Default destructor
    ~Node() = default;

    // Get the average value of the node
    float get_value() const {
        return visit_count == 0 ? 0.0 : value_sum / visit_count;
    }

    // Update the node's visit count and value sum
    void update(float value) {
        visit_count++;
        value_sum += value;
    }

    // Recursively update the node and its parent node's values
    void update_recursive(float leaf_value, const std::string& battle_mode_in_simulation_env) {
        // Pass strings as const references to avoid copying, improve efficiency,
        // and ensure the function cannot modify the original string.
        if (battle_mode_in_simulation_env == "self_play_mode") {
            // Self-play mode: update the current node and recursively update the parent node
            // (pass the negative leaf_value).
            update(leaf_value);
            if (!is_root()) {
                parent->update_recursive(-leaf_value, battle_mode_in_simulation_env);
            }
        }
        else if (battle_mode_in_simulation_env == "play_with_bot_mode") {
            // Play-with-bot mode: update the current node and recursively update the parent node
            // (pass the same leaf_value).
            update(leaf_value);
            if (!is_root()) {
                parent->update_recursive(leaf_value, battle_mode_in_simulation_env);
            }
        }
        else {
            // Error handling: an invalid battle_mode_in_simulation_env was provided.
            std::cerr << "Error: Invalid battle mode '" << battle_mode_in_simulation_env
                      << "' provided to update_recursive()." << std::endl;
            return;
        }
    }

    // Check if the node is a leaf node
    bool is_leaf() const {
        return children.empty();
    }

    // Check if the node is the root node
    bool is_root() const {
        return parent == nullptr;
    }

    // Add a child node
    void add_child(int action, std::shared_ptr<Node> node) {
        children[action] = node;
    }

    // Get the visit count
    int get_visit_count() const { return visit_count; }

    // Get the parent node
    std::shared_ptr<Node> get_parent() const {
        return parent;
    }

    // Get the child nodes
    const std::map<int, std::shared_ptr<Node>>& get_children() const {
        return children;
    }

    float prior_p;        // The prior probability of the node
    int visit_count;      // Visit count
    float value_sum;      // Value sum
};

#endif // NODE_ALPHAZERO_H