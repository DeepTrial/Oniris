/**
 * @file onnx_matcher_style.hpp
 * @brief ONNX Matcher style pattern matching (inspired by onnx_matcher)
 * 
 * Pattern syntax:
 *   OpType(input_tensors, output_tensors)
 *   OpType1/OpType2(input, output)
 *   ?(input, output)  // wildcard for any op type
 * 
 * Examples:
 *   Conv(?, c0)           # Conv with any input, output named c0
 *   Sigmoid(c0, s0)       # Sigmoid takes c0, outputs s0
 *   Mul([s0, c0], ?)      # Mul with two inputs s0 and c0
 *   Conv/Pool(?, c0)      # Conv or Pool
 */

#pragma once

#include "ir/graph.hpp"
#include "ir/model.hpp"
#include "ir/node.hpp"


#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace oniris {
namespace passes {

/**
 * @brief Tensor reference in pattern (variable name or wildcard)
 */
struct TensorRef {
    std::string name;  // Variable name (e.g., "c0") or "?" for wildcard
    bool is_wildcard = false;
    bool is_list = false;  // For [a, b, c] style multi-tensor
    std::vector<TensorRef> list_items;  // If is_list is true
    
    TensorRef() = default;
    explicit TensorRef(std::string n) {
        is_wildcard = (n == "?");
        name = std::move(n);
    }
    
    static TensorRef Wildcard() { return TensorRef("?"); }
    static TensorRef Variable(std::string name) { return TensorRef(name); }
};

/**
 * @brief Match result for subgraph pattern matching
 */
struct SubgraphMatch {
    std::vector<std::shared_ptr<Node>> matched_nodes;
    std::unordered_map<std::string, std::shared_ptr<Node>> node_mapping;
    
    bool IsValid() const { return !matched_nodes.empty(); }
};

/**
 * @brief Node pattern in onnx_matcher style
 */
struct OnnxMatcherNodePattern {
    std::vector<std::string> op_types;  // Can be multiple: ["Conv", "Pool"]
    std::vector<TensorRef> inputs;
    std::vector<TensorRef> outputs;
    
    bool MatchesOpType(const std::string& op_type) const {
        for (const auto& allowed : op_types) {
            if (allowed == "?" || allowed == op_type) {
                return true;
            }
        }
        return false;
    }
};

/**
 * @brief Complete pattern in onnx_matcher style
 */
class OnnxMatcherPattern {
public:
    std::vector<OnnxMatcherNodePattern> nodes;
    
    // Parse from string
    static std::optional<OnnxMatcherPattern> FromString(const std::string& pattern_str);
    
private:
    static bool ParseNodeLine(const std::string& line, OnnxMatcherNodePattern& out);
    static bool ParseTensorRef(const std::string& str, TensorRef& out);
};

/**
 * @brief Matcher for onnx_matcher style patterns
 */
class OnnxMatcherStyleMatcher {
public:
    /**
     * @brief Find all matches of a pattern in a model
     */
    static std::vector<SubgraphMatch> FindAll(
        const std::shared_ptr<Model>& model,
        const OnnxMatcherPattern& pattern);
    
    /**
     * @brief Find first match
     */
    static SubgraphMatch FindFirst(
        const std::shared_ptr<Model>& model,
        const OnnxMatcherPattern& pattern);
    
    /**
     * @brief Check if pattern exists
     */
    static bool HasMatch(
        const std::shared_ptr<Model>& model,
        const OnnxMatcherPattern& pattern);

private:
    static std::vector<std::vector<std::shared_ptr<Node>>> TryMatch(
        const std::shared_ptr<Model>& model,
        const OnnxMatcherPattern& pattern,
        std::shared_ptr<Node> anchor);
};

}  // namespace passes
}  // namespace oniris
