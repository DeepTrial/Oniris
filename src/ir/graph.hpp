/**
 * @file graph.hpp
 * @brief Graph definition containing nodes and tensors
 */

#pragma once

#include "ir/node.hpp"
#include "ir/tensor.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace oniris {

/**
 * @brief Computation graph
 */
class Graph : public std::enable_shared_from_this<Graph> {
public:
    Graph() = default;
    
    /**
     * @brief Create a graph with name
     * @param name Graph name
     */
    explicit Graph(std::string name) : name_(std::move(name)) {}
    
    /// Get graph name
    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }
    
    /// Node management
    const std::vector<std::shared_ptr<Node>>& GetNodes() const { return nodes_; }
    std::vector<std::shared_ptr<Node>>& GetNodes() { return nodes_; }
    
    /**
     * @brief Add a node to the graph
     * @param node Node to add
     * @return Pointer to the added node
     */
    std::shared_ptr<Node> AddNode(std::shared_ptr<Node> node);
    
    /**
     * @brief Create and add a new node
     * @param op_type Operator type
     * @param name Node name
     * @return Pointer to the created node
     */
    std::shared_ptr<Node> CreateNode(const std::string& op_type, const std::string& name = "");
    
    /**
     * @brief Remove a node from the graph
     * @param node Node to remove
     * @return true if removed successfully
     */
    bool RemoveNode(const std::shared_ptr<Node>& node);
    
    /**
     * @brief Remove nodes that are not connected to outputs
     */
    void RemoveDeadNodes();
    
    /// Input/Output value management
    const std::vector<ValueInfo>& GetInputs() const { return inputs_; }
    std::vector<ValueInfo>& GetInputs() { return inputs_; }
    void AddInput(const ValueInfo& info) { inputs_.push_back(info); }
    void SetInputs(const std::vector<ValueInfo>& inputs) { inputs_ = inputs; }
    
    const std::vector<ValueInfo>& GetOutputs() const { return outputs_; }
    std::vector<ValueInfo>& GetOutputs() { return outputs_; }
    void AddOutput(const ValueInfo& info) { outputs_.push_back(info); }
    void SetOutputs(const std::vector<ValueInfo>& outputs) { outputs_ = outputs; }
    
    /// Value info management (for intermediate values)
    bool HasValueInfo(const std::string& name) const {
        return value_infos_.find(name) != value_infos_.end();
    }
    
    const ValueInfo* GetValueInfo(const std::string& name) const {
        auto it = value_infos_.find(name);
        if (it != value_infos_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    ValueInfo* GetValueInfo(const std::string& name) {
        auto it = value_infos_.find(name);
        if (it != value_infos_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    void SetValueInfo(const std::string& name, const ValueInfo& info) {
        value_infos_[name] = info;
    }
    
    void RemoveValueInfo(const std::string& name) {
        value_infos_.erase(name);
    }
    
    /// Get all value infos
    const std::unordered_map<std::string, ValueInfo>& GetAllValueInfos() const {
        return value_infos_;
    }
    
    /// Constant tensor management
    bool HasConstant(const std::string& name) const {
        return constants_.find(name) != constants_.end();
    }
    
    const ConstantTensor* GetConstant(const std::string& name) const {
        auto it = constants_.find(name);
        if (it != constants_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    ConstantTensor* GetConstant(const std::string& name) {
        auto it = constants_.find(name);
        if (it != constants_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    void AddConstant(const std::string& name, const ConstantTensor& constant) {
        constants_[name] = constant;
    }
    
    void RemoveConstant(const std::string& name) {
        constants_.erase(name);
    }
    
    const std::unordered_map<std::string, ConstantTensor>& GetConstants() const {
        return constants_;
    }
    
    /// Initializer management (ONNX-style constants)
    const std::unordered_map<std::string, Tensor>& GetInitializers() const {
        return initializers_;
    }
    
    void AddInitializer(const std::string& name, const Tensor& tensor) {
        initializers_[name] = tensor;
    }
    
    /// Get all value names in the graph
    std::vector<std::string> GetAllValueNames() const;
    
    /// Find producer node of a value
    std::shared_ptr<Node> GetProducer(const std::string& value_name) const;
    
    /// Find consumer nodes of a value
    std::vector<std::shared_ptr<Node>> GetConsumers(const std::string& value_name) const;
    
    /// Topological sort of nodes
    std::vector<std::shared_ptr<Node>> TopologicalSort() const;
    
    /// Clone the graph
    std::shared_ptr<Graph> Clone() const;
    
    /// Validate graph structure
    bool Validate(std::string* error_msg = nullptr) const;

private:
    std::string name_;
    std::vector<std::shared_ptr<Node>> nodes_;
    std::vector<ValueInfo> inputs_;
    std::vector<ValueInfo> outputs_;
    std::unordered_map<std::string, ValueInfo> value_infos_;
    std::unordered_map<std::string, ConstantTensor> constants_;
    std::unordered_map<std::string, Tensor> initializers_;
    
    /// Build value name to producer mapping
    std::unordered_map<std::string, std::shared_ptr<Node>> BuildProducerMap() const;
};

}  // namespace oniris
