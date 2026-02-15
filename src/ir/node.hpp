/**
 * @file node.hpp
 * @brief Node (operation) definition
 */

#pragma once

#include "core/types.hpp"
#include "ir/tensor.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace oniris {

/**
 * @brief Attribute value types
 */
using AttributeValue = std::variant<
    int64_t,                    // INT
    float,                      // FLOAT
    std::string,                // STRING
    std::vector<int64_t>,       // INTS
    std::vector<float>,         // FLOATS
    std::vector<std::string>,   // STRINGS
    Tensor                      // TENSOR
>;

/**
 * @brief Node represents an operation in the graph
 */
class Node {
public:
    Node() = default;
    
    /**
     * @brief Create a node
     * @param op_type Operator type (e.g., "Conv", "MatMul")
     * @param name Optional node name
     */
    Node(std::string op_type, std::string name = "")
        : op_type_(std::move(op_type)), name_(std::move(name)) {}
    
    /// Get operator type
    const std::string& GetOpType() const { return op_type_; }
    void SetOpType(const std::string& op_type) { op_type_ = op_type; }
    
    /// Get node name
    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }
    
    /// Get/set domain
    const std::string& GetDomain() const { return domain_; }
    void SetDomain(const std::string& domain) { domain_ = domain; }
    
    /// Input/output management
    const std::vector<std::string>& GetInputs() const { return inputs_; }
    std::vector<std::string>& GetInputs() { return inputs_; }
    void AddInput(const std::string& name) { inputs_.push_back(name); }
    void SetInput(size_t idx, const std::string& name) { inputs_[idx] = name; }
    void ClearInputs() { inputs_.clear(); }
    
    const std::vector<std::string>& GetOutputs() const { return outputs_; }
    std::vector<std::string>& GetOutputs() { return outputs_; }
    void AddOutput(const std::string& name) { outputs_.push_back(name); }
    void ClearOutputs() { outputs_.clear(); }
    
    /// Attribute management
    bool HasAttribute(const std::string& name) const {
        return attributes_.find(name) != attributes_.end();
    }
    
    const AttributeValue* GetAttribute(const std::string& name) const {
        auto it = attributes_.find(name);
        if (it != attributes_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    template<typename T>
    std::optional<T> GetAttributeAs(const std::string& name) const {
        auto it = attributes_.find(name);
        if (it != attributes_.end()) {
            if (std::holds_alternative<T>(it->second)) {
                return std::get<T>(it->second);
            }
        }
        return std::nullopt;
    }
    
    void SetAttribute(const std::string& name, AttributeValue value) {
        attributes_[name] = std::move(value);
    }
    
    void RemoveAttribute(const std::string& name) {
        attributes_.erase(name);
    }
    
    const std::unordered_map<std::string, AttributeValue>& GetAttributes() const {
        return attributes_;
    }
    
    /// Get output shapes (set by shape inference)
    const std::vector<Shape>& GetOutputShapes() const { return output_shapes_; }
    std::vector<Shape>& GetOutputShapes() { return output_shapes_; }
    void SetOutputShape(size_t idx, const Shape& shape);
    
    /// Check if all output shapes are inferred
    bool HasInferredShapes() const;
    
    /// Clone the node
    std::shared_ptr<Node> Clone() const;

private:
    std::string op_type_;
    std::string name_;
    std::string domain_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, AttributeValue> attributes_;
    std::vector<Shape> output_shapes_;
};

}  // namespace oniris
