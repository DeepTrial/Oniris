/**
 * @file onnx_matcher_style.cpp
 * @brief ONNX Matcher style pattern matching implementation
 */

#include "passes/onnx_matcher_style.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <regex>
#include <sstream>

namespace oniris {
namespace passes {

// ============================================================================
// Utility Functions
// ============================================================================

static std::string Trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, last - first + 1);
}

// ============================================================================
// Parsing
// ============================================================================

bool OnnxMatcherPattern::ParseTensorRef(const std::string& str, TensorRef& out) {
    std::string trimmed = Trim(str);
    
    if (trimmed.empty()) {
        return false;
    }
    
    // Check for list: [a, b, c]
    if (trimmed.size() >= 2 && trimmed[0] == '[' && trimmed[trimmed.size()-1] == ']') {
        out.is_list = true;
        out.name = "[]";
        
        std::string inner = trimmed.substr(1, trimmed.size() - 2);
        std::stringstream ss(inner);
        std::string item;
        
        while (std::getline(ss, item, ',')) {
            TensorRef item_ref;
            if (ParseTensorRef(item, item_ref)) {
                out.list_items.push_back(item_ref);
            }
        }
        
        return !out.list_items.empty();
    }
    
    // Single variable
    out.name = trimmed;
    out.is_wildcard = (trimmed == "?");
    return true;
}

bool OnnxMatcherPattern::ParseNodeLine(const std::string& line, OnnxMatcherNodePattern& out) {
    // Format: OpType(input_tensors, output_tensors)
    // Or: OpType1/OpType2(input_tensors, output_tensors)
    // Examples:
    //   Conv(?, c0)
    //   Sigmoid(c0, s0)
    //   Mul([s0, c0], ?)
    
    // Remove spaces
    std::string clean_line = line;
    clean_line.erase(std::remove_if(clean_line.begin(), clean_line.end(), ::isspace), clean_line.end());
    
    // Match: Name(Arguments)
    std::regex node_regex(R"((\w+(?:/\w+)*|\?)\(([^)]*)\))");
    std::smatch match;
    
    if (!std::regex_match(clean_line, match, node_regex)) {
        return false;
    }
    
    std::string op_types_str = match[1].str();
    std::string args_str = match[2].str();
    
    // Parse op types (can be multiple separated by /)
    std::stringstream op_ss(op_types_str);
    std::string op_type;
    while (std::getline(op_ss, op_type, '/')) {
        out.op_types.push_back(op_type);
    }
    
    // Parse arguments - split by comma, but respect brackets
    std::vector<std::string> arg_parts;
    std::string current_arg;
    int bracket_depth = 0;
    
    for (char c : args_str) {
        if (c == '[') bracket_depth++;
        else if (c == ']') bracket_depth--;
        
        if (c == ',' && bracket_depth == 0) {
            arg_parts.push_back(current_arg);
            current_arg.clear();
        } else {
            current_arg += c;
        }
    }
    if (!current_arg.empty()) {
        arg_parts.push_back(current_arg);
    }
    
    // Should have exactly 2 parts: inputs and outputs
    if (arg_parts.size() != 2) {
        return false;
    }
    
    // Parse inputs
    TensorRef input_ref;
    if (ParseTensorRef(arg_parts[0], input_ref)) {
        if (input_ref.is_list) {
            out.inputs = input_ref.list_items;
        } else {
            out.inputs.push_back(input_ref);
        }
    }
    
    // Parse outputs
    TensorRef output_ref;
    if (ParseTensorRef(arg_parts[1], output_ref)) {
        if (output_ref.is_list) {
            out.outputs = output_ref.list_items;
        } else {
            out.outputs.push_back(output_ref);
        }
    }
    
    return true;
}

std::optional<OnnxMatcherPattern> OnnxMatcherPattern::FromString(const std::string& pattern_str) {
    OnnxMatcherPattern pattern;
    
    std::istringstream stream(pattern_str);
    std::string line;
    
    while (std::getline(stream, line)) {
        line = Trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Remove inline comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
            line = Trim(line);
        }
        
        // Skip if line becomes empty after removing comments
        if (line.empty()) {
            continue;
        }
        
        OnnxMatcherNodePattern node;
        if (ParseNodeLine(line, node)) {
            pattern.nodes.push_back(node);
        } else {
            ONIRIS_WARNING << "Failed to parse pattern line: " << line;
        }
    }
    
    if (pattern.nodes.empty()) {
        return std::nullopt;
    }
    
    return pattern;
}

// ============================================================================
// Matching
// ============================================================================

static bool MatchTensorRef(const TensorRef& ref, const std::string& actual_tensor,
                           std::unordered_map<std::string, std::string>& variables) {
    if (ref.is_wildcard) {
        return true;
    }
    
    // Check if we already have a binding for this variable
    auto it = variables.find(ref.name);
    if (it != variables.end()) {
        return it->second == actual_tensor;
    }
    
    // New binding
    variables[ref.name] = actual_tensor;
    return true;
}

static bool CheckInputs(const std::vector<TensorRef>& pattern_inputs,
                        const std::vector<std::string>& node_inputs,
                        std::unordered_map<std::string, std::string>& variables) {
    // Check counts match (unless pattern has wildcards)
    size_t pattern_count = pattern_inputs.size();
    size_t actual_count = node_inputs.size();
    
    // Allow partial matching if last input is wildcard (for optional inputs)
    if (pattern_count > actual_count) {
        return false;
    }
    
    for (size_t i = 0; i < pattern_count; ++i) {
        if (!MatchTensorRef(pattern_inputs[i], node_inputs[i], variables)) {
            return false;
        }
    }
    
    return true;
}

static bool CheckOutputs(const std::vector<TensorRef>& pattern_outputs,
                         const std::vector<std::string>& node_outputs,
                         std::unordered_map<std::string, std::string>& variables) {
    if (pattern_outputs.size() != node_outputs.size()) {
        return false;
    }
    
    for (size_t i = 0; i < pattern_outputs.size(); ++i) {
        if (!MatchTensorRef(pattern_outputs[i], node_outputs[i], variables)) {
            return false;
        }
    }
    
    return true;
}

std::vector<std::vector<std::shared_ptr<Node>>> OnnxMatcherStyleMatcher::TryMatch(
    const std::shared_ptr<Model>& model,
    const OnnxMatcherPattern& pattern,
    std::shared_ptr<Node> anchor) {
    
    std::vector<std::vector<std::shared_ptr<Node>>> results;
    
    if (!model || !anchor || pattern.nodes.empty()) {
        return results;
    }
    
    auto graph = model->GetGraph();
    if (!graph) {
        return results;
    }
    
    // Stack-based DFS: [path, pattern_index, variables]
    struct StackItem {
        std::vector<std::shared_ptr<Node>> path;
        size_t pattern_idx;
        std::unordered_map<std::string, std::string> variables;
    };
    
    std::vector<StackItem> stack;
    
    // Check if anchor matches first pattern node
    const auto& first_pattern = pattern.nodes[0];
    if (!first_pattern.MatchesOpType(anchor->GetOpType())) {
        return results;
    }
    
    std::unordered_map<std::string, std::string> initial_vars;
    if (!CheckInputs(first_pattern.inputs, anchor->GetInputs(), initial_vars)) {
        return results;
    }
    if (!CheckOutputs(first_pattern.outputs, anchor->GetOutputs(), initial_vars)) {
        return results;
    }
    
    stack.push_back({{anchor}, 0, initial_vars});
    
    while (!stack.empty()) {
        auto item = stack.back();
        stack.pop_back();
        
        auto current_node = item.path.back();
        size_t pattern_idx = item.pattern_idx;
        auto variables = item.variables;
        
        // Check current node matches pattern
        const auto& node_pattern = pattern.nodes[pattern_idx];
        if (!node_pattern.MatchesOpType(current_node->GetOpType())) {
            continue;
        }
        
        // Check inputs (using accumulated variables)
        if (!CheckInputs(node_pattern.inputs, current_node->GetInputs(), variables)) {
            continue;
        }
        
        // Check outputs
        if (!CheckOutputs(node_pattern.outputs, current_node->GetOutputs(), variables)) {
            continue;
        }
        
        // If this is the last pattern node, we have a match
        if (pattern_idx == pattern.nodes.size() - 1) {
            results.push_back(item.path);
            continue;
        }
        
        // Continue to next pattern node
        size_t next_idx = pattern_idx + 1;
        const auto& next_pattern = pattern.nodes[next_idx];
        
        // Get expected inputs for next node from pattern
        std::vector<std::string> expected_inputs;
        for (const auto& input_ref : next_pattern.inputs) {
            if (!input_ref.is_wildcard && !input_ref.name.empty()) {
                auto it = variables.find(input_ref.name);
                if (it != variables.end()) {
                    expected_inputs.push_back(it->second);
                }
            }
        }
        
        // Find candidates for next node
        for (const auto& output_name : current_node->GetOutputs()) {
            auto consumers = graph->GetConsumers(output_name);
            for (const auto& consumer : consumers) {
                // Check if this consumer is already in path
                if (std::find(item.path.begin(), item.path.end(), consumer) != item.path.end()) {
                    continue;
                }
                
                // For multi-branch patterns, we need to check if inputs match
                bool inputs_match = true;
                if (!expected_inputs.empty()) {
                    const auto& consumer_inputs = consumer->GetInputs();
                    for (const auto& expected : expected_inputs) {
                        if (std::find(consumer_inputs.begin(), consumer_inputs.end(), expected) == consumer_inputs.end()) {
                            inputs_match = false;
                            break;
                        }
                    }
                }
                
                if (inputs_match) {
                    auto new_path = item.path;
                    new_path.push_back(consumer);
                    stack.push_back({new_path, next_idx, variables});
                }
            }
        }
    }
    
    return results;
}

std::vector<SubgraphMatch> OnnxMatcherStyleMatcher::FindAll(
    const std::shared_ptr<Model>& model,
    const OnnxMatcherPattern& pattern) {
    
    std::vector<SubgraphMatch> matches;
    
    if (!model || pattern.nodes.empty()) {
        return matches;
    }
    
    auto graph = model->GetGraph();
    if (!graph) {
        return matches;
    }
    
    // Try each node as anchor
    for (const auto& node : graph->GetNodes()) {
        if (node->GetOpType() == "Constant") {
            continue;
        }
        
        auto paths = TryMatch(model, pattern, node);
        for (const auto& path : paths) {
            SubgraphMatch match;
            match.matched_nodes = path;
            
            // Generate node names for mapping
            for (size_t i = 0; i < path.size() && i < pattern.nodes.size(); ++i) {
                std::string name = "node" + std::to_string(i);
                match.node_mapping[name] = path[i];
            }
            
            matches.push_back(match);
        }
    }
    
    // Remove duplicates
    std::sort(matches.begin(), matches.end(), [](const SubgraphMatch& a, const SubgraphMatch& b) {
        if (a.matched_nodes.size() != b.matched_nodes.size()) {
            return a.matched_nodes.size() < b.matched_nodes.size();
        }
        for (size_t i = 0; i < a.matched_nodes.size(); ++i) {
            if (a.matched_nodes[i].get() != b.matched_nodes[i].get()) {
                return a.matched_nodes[i].get() < b.matched_nodes[i].get();
            }
        }
        return false;
    });
    
    matches.erase(std::unique(matches.begin(), matches.end(), 
        [](const SubgraphMatch& a, const SubgraphMatch& b) {
            if (a.matched_nodes.size() != b.matched_nodes.size()) return false;
            for (size_t i = 0; i < a.matched_nodes.size(); ++i) {
                if (a.matched_nodes[i].get() != b.matched_nodes[i].get()) {
                    return false;
                }
            }
            return true;
        }), matches.end());
    
    return matches;
}

SubgraphMatch OnnxMatcherStyleMatcher::FindFirst(
    const std::shared_ptr<Model>& model,
    const OnnxMatcherPattern& pattern) {
    
    auto matches = FindAll(model, pattern);
    if (!matches.empty()) {
        return matches[0];
    }
    return SubgraphMatch();
}

bool OnnxMatcherStyleMatcher::HasMatch(
    const std::shared_ptr<Model>& model,
    const OnnxMatcherPattern& pattern) {
    
    return !FindAll(model, pattern).empty();
}

}  // namespace passes
}  // namespace oniris
