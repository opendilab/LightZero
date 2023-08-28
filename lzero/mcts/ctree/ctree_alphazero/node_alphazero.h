#include <map>
#include <string>
#include <iostream>
#include <memory>
#include <mutex>

class Node {
    //  std::mutex mtx;  // 互斥锁
    //   std::recursive_mutex mtx;  // 递归互斥锁
public:
    Node(Node* parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value_sum(0.0) {}

    ~Node() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
//
//    void remove_from_parent() {
//    if (parent != nullptr) {
//        parent->children.erase(std::find_if(parent->children.begin(), parent->children.end(),
//            [this](const std::pair<int, Node*>& pair) { return pair.second == this; }));
//    }
//    }
//
//    void end_game(Node* root) {
//        // 假设你在游戏结束时不再需要树中的所有节点
//        delete_subtree(root);
//    }
//
//    void delete_subtree(Node* node) {
//        printf("position-ds-1 \n");
//
//        for (auto& pair : node->children) {
//            delete_subtree(pair.second);
//        }
//        printf("position-ds-2 \n");
//
//        node->remove_from_parent();
//        printf("position-ds-3 \n");
//
//        delete node;
//        printf("position-ds-4 \n");
//
//    }


    // node->remove_from_parent();
    // delete node;

    float get_value() {
        return visit_count == 0 ? 0.0 : value_sum / visit_count;
    }

    void update(float value) {
        visit_count++;
        value_sum += value;
    }

    void update_recursive(float leaf_value, std::string mcts_mode) {
        
        // printf("parent pointer: %p\n", parent);
        // printf("position-ur-1 \n");
        
        // std::lock_guard<std::mutex> lock(mtx);  // 自动获取锁
        // std::lock_guard<std::recursive_mutex> lock(mtx);  // 自动获取锁
        // 在 lock_guard 对象析构时自动释放锁


        if (mcts_mode == "self_play_mode") {
            // printf("position-ur-2 \n");
            // printf("leaf_value: %f\n", leaf_value);

            update(leaf_value);
            
            // printf("position-ur-3 \n");
            if (!is_root()) {
                // printf("position-ur-4  \n");
                parent->update_recursive(-leaf_value, mcts_mode);
                // printf("position-ur-5  \n");
            }
            // printf("position-ur-6 \n");
        }
        else if (mcts_mode == "play_with_bot_mode") {
            update(leaf_value);
            if (!is_root()) {
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

    // std::map<int, std::unique_ptr<Node>> get_children() {
    //     return children;
    // }


    int get_visit_count() {
        return visit_count;
    }

    void add_child(int action, Node* node) {
        children[action] = node;
    }
    // void add_child(int action, std::unique_ptr<Node> node) {
    //     children[action] = std::move(node);
    // }

public:
    Node* parent;
    float prior_p;
    int visit_count;
    float value_sum;
    std::map<int, Node*> children;
    // std::map<int, std::unique_ptr<Node>> children;

};