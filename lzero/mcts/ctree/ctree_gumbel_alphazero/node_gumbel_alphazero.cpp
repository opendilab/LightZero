#include "node_gumbel_alphazero.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(node_gumbel_alphazero, m) {
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
        .def("children", &Node::get_children)
        .def_readwrite("children", &Node::children)
        .def("add_child", &Node::add_child)
        .def("visit_count", &Node::get_visit_count);
}


// 构建与编译命令
// mkdir build
// cd buld
// cmake ..
// make