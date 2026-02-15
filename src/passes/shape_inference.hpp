/**
 * @file shape_inference.hpp
 * @brief Shape inference engine with support for custom layers
 */

#pragma once

#include "ir/graph.hpp"
#include "ir/model.hpp"

#include <functional>
#include <memory>
#include <unordered_map>

namespace oniris {
namespace passes {

/**
 * @brief Context for shape inference
 */
struct InferenceContext {
    /// Input shapes to the node
    std::vector<Shape> input_shapes;
    
    /// Input data types
    std::vector<DataType> input_dtypes;
    
    /// Node attributes
    const std::unordered_map<std::string, AttributeValue>* attributes = nullptr;
    
    /// Graph-level information
    const Graph* graph = nullptr;
    
    /// Get input shape at index
    const Shape& GetInputShape(size_t idx) const {
        static Shape empty;
        if (idx < input_shapes.size()) {
            return input_shapes[idx];
        }
        return empty;
    }
    
    /// Get input dtype at index
    DataType GetInputDtype(size_t idx) const {
        if (idx < input_dtypes.size()) {
            return input_dtypes[idx];
        }
        return DataType::kUnknown;
    }
    
    /// Get attribute value
    template<typename T>
    std::optional<T> GetAttribute(const std::string& name) const {
        if (!attributes) return std::nullopt;
        auto it = attributes->find(name);
        if (it != attributes->end()) {
            if (std::holds_alternative<T>(it->second)) {
                return std::get<T>(it->second);
            }
        }
        return std::nullopt;
    }
};

/**
 * @brief Result of shape inference
 */
struct InferenceResult {
    /// Output shapes (one per output)
    std::vector<Shape> output_shapes;
    /// Output data types (optional)
    std::vector<DataType> output_dtypes;
    /// Whether inference succeeded
    bool success = true;
    /// Error message if failed
    std::string error_msg;
    
    static InferenceResult Error(const std::string& msg) {
        InferenceResult result;
        result.success = false;
        result.error_msg = msg;
        return result;
    }
    
    static InferenceResult Success(std::vector<Shape> shapes) {
        InferenceResult result;
        result.output_shapes = std::move(shapes);
        return result;
    }
};

/**
 * @brief Shape inference function type
 */
using ShapeInferFunc = std::function<InferenceResult(const InferenceContext&)>;

/**
 * @brief Shape inference engine with plugin support
 */
class ShapeInferenceEngine {
public:
    /**
     * @brief Get the global shape inference engine instance
     * @return Reference to the global engine
     */
    static ShapeInferenceEngine& GetInstance();
    
    /**
     * @brief Register a shape inference function for an operator
     * @param op_type Operator type name
     * @param func Inference function
     * @param domain Operator domain (default: "")
     */
    void Register(const std::string& op_type, ShapeInferFunc func, 
                  const std::string& domain = "");
    
    /**
     * @brief Unregister a shape inference function
     * @param op_type Operator type name
     * @param domain Operator domain
     */
    void Unregister(const std::string& op_type, const std::string& domain = "");
    
    /**
     * @brief Check if an operator has a registered inference function
     * @param op_type Operator type name
     * @param domain Operator domain
     * @return true if registered
     */
    bool HasHandler(const std::string& op_type, const std::string& domain = "") const;
    
    /**
     * @brief Infer shapes for a single node
     * @param node The node to infer
     * @param graph The containing graph (for context)
 * @return Inference result
     */
    InferenceResult InferNode(const std::shared_ptr<Node>& node, const Graph& graph);
    
    /**
     * @brief Run shape inference on the entire graph
     * @param graph The graph to process
     * @param fail_on_unknown If false, skip unsupported nodes without error
     * @return true if all nodes were successfully processed
     */
    bool InferGraph(const std::shared_ptr<Graph>& graph, bool fail_on_unknown = false);
    
    /**
     * @brief Run shape inference on a model
     * @param model The model to process
     * @param fail_on_unknown If false, skip unsupported nodes without error
     * @return true if all nodes were successfully processed
     */
    bool InferModel(const std::shared_ptr<Model>& model, bool fail_on_unknown = false);
    
    /// Get list of supported operators
    std::vector<std::string> GetSupportedOps() const;

private:
    ShapeInferenceEngine();
    ShapeInferenceEngine(const ShapeInferenceEngine&) = delete;
    ShapeInferenceEngine& operator=(const ShapeInferenceEngine&) = delete;
    
public:
    ~ShapeInferenceEngine() = default;
    
    // Key: "domain::op_type"
    std::unordered_map<std::string, ShapeInferFunc> handlers_;
    
    // Build key from domain and op_type
    static std::string MakeKey(const std::string& op_type, const std::string& domain);
    
    // Initialize default handlers
    void InitializeDefaultHandlers();
    
    // Helper to get value info from graph
    static bool GetValueInfo(const Graph& graph, const std::string& name,
                            Shape* shape, DataType* dtype);
};

/**
 * @brief RAII helper for registering custom handlers
 */
class CustomHandlerRegistrar {
public:
    CustomHandlerRegistrar(const std::string& op_type, ShapeInferFunc func,
                           const std::string& domain = "") {
        ShapeInferenceEngine::GetInstance().Register(op_type, func, domain);
    }
};

/**
 * @brief Macro for easy registration of custom handlers
 */
#define ONIRIS_REGISTER_SHAPE_INFER(op_type, func) \
    static ::oniris::passes::CustomHandlerRegistrar \
        _oniris_shape_infer_##op_type##_##func(#op_type, func)

}  // namespace passes
}  // namespace oniris
