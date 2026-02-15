/**
 * @file graph.cpp
 * @brief Graph implementation
 */

#include "ir/graph.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace oniris {

std::shared_ptr<Node> Graph::AddNode(std::shared_ptr<Node> node) {
    nodes_.push_back(node);
    return node;
}

std::shared_ptr<Node> Graph::CreateNode(const std::string& op_type, const std::string& name) {
    auto node = std::make_shared<Node>(op_type, name);
    return AddNode(node);
}

bool Graph::RemoveNode(const std::shared_ptr<Node>& node) {
    auto it = std::find(nodes_.begin(), nodes_.end(), node);
    if (it != nodes_.end()) {
        nodes_.erase(it);
        return true;
    }
    return false;
}

void Graph::RemoveDeadNodes() {
    // Build set of values needed by outputs
    std::unordered_set<std::string> needed_values;
    for (const auto& output : outputs_) {
        needed_values.insert(output.name);
    }
    
    // Work backwards through nodes
    std::vector<std::shared_ptr<Node>> alive_nodes;
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
        const auto& node = *it;
        bool needed = false;
        
        for (const auto& output : node->GetOutputs()) {
            if (needed_values.find(output) != needed_values.end()) {
                needed = true;
                break;
            }
        }
        
        if (needed) {
            alive_nodes.push_back(node);
            // Add inputs as needed
            for (const auto& input : node->GetInputs()) {
                needed_values.insert(input);
            }
        }
    }
    
    // Reverse to restore original order
    std::reverse(alive_nodes.begin(), alive_nodes.end());
    nodes_ = std::move(alive_nodes);
}

std::vector<std::string> Graph::GetAllValueNames() const {
    std::unordered_set<std::string> names;
    
    // Add inputs
    for (const auto& input : inputs_) {
        names.insert(input.name);
    }
    
    // Add outputs
    for (const auto& output : outputs_) {
        names.insert(output.name);
    }
    
    // Add all node outputs
    for (const auto& node : nodes_) {
        for (const auto& output : node->GetOutputs()) {
            names.insert(output);
        }
    }
    
    return std::vector<std::string>(names.begin(), names.end());
}

std::unordered_map<std::string, std::shared_ptr<Node>> Graph::BuildProducerMap() const {
    std::unordered_map<std::string, std::shared_ptr<Node>> producers;
    for (const auto& node : nodes_) {
        for (const auto& output : node->GetOutputs()) {
            producers[output] = node;
        }
    }
    return producers;
}

std::shared_ptr<Node> Graph::GetProducer(const std::string& value_name) const {
    auto producers = BuildProducerMap();
    auto it = producers.find(value_name);
    if (it != producers.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Node>> Graph::GetConsumers(const std::string& value_name) const {
    std::vector<std::shared_ptr<Node>> consumers;
    for (const auto& node : nodes_) {
        for (const auto& input : node->GetInputs()) {
            if (input == value_name) {
                consumers.push_back(node);
                break;
            }
        }
    }
    return consumers;
}

std::vector<std::shared_ptr<Node>> Graph::TopologicalSort() const {
    std::unordered_map<std::string, std::shared_ptr<Node>> producers = BuildProducerMap();
    std::unordered_map<std::shared_ptr<Node>, int> in_degree;
    
    // Calculate in-degree for each node
    for (const auto& node : nodes_) {
        int degree = 0;
        for (const auto& input : node->GetInputs()) {
            // Check if input is produced by another node (not a graph input or initializer)
            if (producers.find(input) != producers.end()) {
                degree++;
            }
        }
        in_degree[node] = degree;
    }
    
    // Kahn's algorithm
    std::queue<std::shared_ptr<Node>> queue;
    for (const auto& node : nodes_) {
        if (in_degree[node] == 0) {
            queue.push(node);
        }
    }
    
    std::vector<std::shared_ptr<Node>> sorted;
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();
        sorted.push_back(node);
        
        // Find all nodes that depend on this node's outputs
        for (const auto& output : node->GetOutputs()) {
            for (const auto& other : nodes_) {
                if (in_degree[other] > 0) {
                    for (const auto& input : other->GetInputs()) {
                        if (input == output) {
                            in_degree[other]--;
                            if (in_degree[other] == 0) {
                                queue.push(other);
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (sorted.size() != nodes_.size()) {
        ONIRIS_WARNING << "Graph contains cycles, topological sort incomplete";
    }
    
    return sorted;
}

std::shared_ptr<Graph> Graph::Clone() const {
    auto cloned = std::make_shared<Graph>(name_);
    cloned->inputs_ = inputs_;
    cloned->outputs_ = outputs_;
    cloned->value_infos_ = value_infos_;
    cloned->constants_ = constants_;
    cloned->initializers_ = initializers_;
    
    for (const auto& node : nodes_) {
        cloned->nodes_.push_back(node->Clone());
    }
    
    return cloned;
}

bool Graph::Validate(std::string* error_msg) const {
    // Check that all node inputs are defined
    std::unordered_set<std::string> available_values;
    for (const auto& input : inputs_) {
        available_values.insert(input.name);
    }
    for (const auto& [name, _] : initializers_) {
        available_values.insert(name);
    }
    
    for (const auto& node : nodes_) {
        for (const auto& input : node->GetInputs()) {
            if (!input.empty() && available_values.find(input) == available_values.end()) {
                if (error_msg) {
                    *error_msg = "Undefined input: " + input + " for node " + node->GetName();
                }
                return false;
            }
        }
        for (const auto& output : node->GetOutputs()) {
            available_values.insert(output);
        }
    }
    
    // Check that all graph outputs are produced
    for (const auto& output : outputs_) {
        if (available_values.find(output.name) == available_values.end()) {
            if (error_msg) {
                *error_msg = "Graph output not produced: " + output.name;
            }
            return false;
        }
    }
    
    return true;
}

}  // namespace oniris
