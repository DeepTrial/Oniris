/**
 * @file compiler.hpp
 * @brief ONNX Model Compiler - Main compilation pipeline
 * 
 * Pipeline: Input ONNX Model -> Optimization Pass -> Shape Inference Pass 
 * -> Pattern Matching -> JSON Output
 */

#pragma once

#include "ir/model.hpp"
#include "passes/simplifier.hpp"
#include "passes/shape_inference.hpp"
#include "passes/onnx_matcher_style.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>

namespace oniris {
namespace passes {

// Forward declarations
struct PatternDefinition;
struct PatternMatchResult;
struct CompilationResult;

/**
 * @brief Pattern match type
 */
enum class PatternMatchType {
    kFirst,     // Stop at first match
    kAll,       // Find all matches
    kCountOnly  // Just count matches
};

/**
 * @brief User-defined pattern for matching
 */
struct PatternDefinition {
    std::string name;                    // Pattern name (for identification)
    std::string pattern_string;          // ONNX matcher style pattern string
    std::optional<OnnxMatcherPattern> parsed_pattern;  // Parsed pattern (computed)
    
    // Optional constraints
    std::unordered_map<std::string, std::string> attributes;  // Attribute constraints
    
    PatternDefinition() = default;
    PatternDefinition(const std::string& n, const std::string& pattern)
        : name(n), pattern_string(pattern) {}
    
    // Parse the pattern string
    bool Parse(std::string* error_msg = nullptr);
};

/**
 * @brief Information about a matched node
 */
struct MatchedNodeInfo {
    std::string node_name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, std::string> attributes;  // Serialized attributes
    std::vector<std::vector<int64_t>> input_shapes;   // Inferred input shapes
    std::vector<std::vector<int64_t>> output_shapes;  // Inferred output shapes
};

/**
 * @brief Single pattern match result
 */
struct PatternMatchResult {
    std::string pattern_name;
    int match_id;  // Index of this match within the pattern
    std::vector<MatchedNodeInfo> nodes;
    std::unordered_map<std::string, std::string> tensor_bindings;  // Variable -> tensor name
    std::unordered_map<std::string, MatchedNodeInfo> node_mapping; // node0, node1, etc.
};

/**
 * @brief Pattern matching results for all patterns
 */
struct PatternMatchingSummary {
    int total_patterns = 0;
    int patterns_with_matches = 0;
    int total_matches = 0;
    std::unordered_map<std::string, std::vector<PatternMatchResult>> pattern_results;
    std::unordered_map<std::string, int> match_counts;  // pattern_name -> count
};

/**
 * @brief Optimization pass statistics
 */
struct OptimizationStats {
    int num_iterations = 0;
    int num_changes = 0;
    std::unordered_map<std::string, int> pass_stats;
    std::vector<std::string> unsupported_ops;
    bool success = true;
    std::string error_msg;
};

/**
 * @brief Shape inference statistics
 */
struct ShapeInferenceStats {
    int num_nodes_processed = 0;
    int num_nodes_failed = 0;
    std::vector<std::string> failed_nodes;
    bool success = true;
    std::string error_msg;
};

/**
 * @brief Model information summary
 */
struct ModelSummary {
    std::string producer_name;
    std::string producer_version;
    int64_t ir_version;
    int64_t opset_version;
    int num_nodes = 0;
    int num_initializers = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<std::string> op_types_used;
    std::unordered_map<std::string, int> op_type_counts;
};

/**
 * @brief Complete compilation result (can be serialized to JSON)
 */
struct CompilationResult {
    bool success = true;
    std::string error_msg;
    std::string input_path;
    std::string output_path;
    
    // Pipeline stages
    ModelSummary model_info;
    OptimizationStats optimization_stats;
    ShapeInferenceStats shape_inference_stats;
    PatternMatchingSummary pattern_matching_summary;
    
    // Timestamp
    std::string start_time;
    std::string end_time;
    double duration_ms = 0;
    
    // Serialize to JSON string
    std::string ToJson(bool pretty = true) const;
    
    // Save JSON to file
    bool SaveJson(const std::string& filepath, bool pretty = true) const;
};

/**
 * @brief Compiler options
 */
struct CompilerOptions {
    // Optimization options
    SimplifyOptions simplify_options;
    bool enable_optimization = true;
    
    // Shape inference options
    bool enable_shape_inference = true;
    bool fail_on_unknown_shape = false;
    
    // Pattern matching options
    bool enable_pattern_matching = true;
    PatternMatchType match_type = PatternMatchType::kAll;
    int max_matches_per_pattern = 1000;  // Limit to avoid memory issues
    bool pattern_match_before_opt = false;  // Run pattern matching before optimization
    
    // Output options
    bool save_optimized_model = true;
    bool save_json_result = true;
    std::string json_output_path;
    
    // Logging
    bool verbose = false;
    std::function<void(const std::string&)> log_callback;
};

/**
 * @brief ONNX Model Compiler
 * 
 * Main compilation pipeline:
 *   1. Load ONNX model
 *   2. Run optimization passes
 *   3. Run shape inference
 *   4. Match user-defined patterns
 *   5. Output JSON results
 */
class ModelCompiler {
public:
    ModelCompiler();
    ~ModelCompiler() = default;
    
    // Disable copy, enable move
    ModelCompiler(const ModelCompiler&) = delete;
    ModelCompiler& operator=(const ModelCompiler&) = delete;
    ModelCompiler(ModelCompiler&&) = default;
    ModelCompiler& operator=(ModelCompiler&&) = default;
    
    /**
     * @brief Add a pattern to match
     * @param pattern The pattern definition
     * @return true if pattern was valid and added
     */
    bool AddPattern(const PatternDefinition& pattern);
    bool AddPattern(const std::string& name, const std::string& pattern_string);
    
    /**
     * @brief Add multiple patterns
     */
    void AddPatterns(const std::vector<PatternDefinition>& patterns);
    
    /**
     * @brief Clear all patterns
     */
    void ClearPatterns();
    
    /**
     * @brief Get number of registered patterns
     */
    size_t GetPatternCount() const { return patterns_.size(); }
    
    /**
     * @brief Get registered pattern names
     */
    std::vector<std::string> GetPatternNames() const;
    
    /**
     * @brief Compile a model file
     * @param input_path Input ONNX model path
     * @param output_path Output optimized model path (optional)
     * @param options Compiler options
     * @return Compilation result
     */
    CompilationResult Compile(
        const std::string& input_path,
        const std::string& output_path = "",
        const CompilerOptions& options = {});
    
    /**
     * @brief Compile a model object
     * @param model The model to compile
     * @param options Compiler options
     * @return Compilation result
     */
    CompilationResult CompileModel(
        const std::shared_ptr<Model>& model,
        const CompilerOptions& options = {});
    
    /**
     * @brief Run only pattern matching on a model
     * @param model The model to match against
     * @return Pattern matching summary
     */
    PatternMatchingSummary RunPatternMatching(
        const std::shared_ptr<Model>& model,
        const PatternMatchType& match_type = PatternMatchType::kAll);

private:
    std::vector<PatternDefinition> patterns_;
    
    // Internal helpers
    void Log(const std::string& msg, const CompilerOptions& options);
    ModelSummary ExtractModelInfo(const std::shared_ptr<Model>& model);
    OptimizationStats RunOptimization(
        const std::shared_ptr<Model>& model, 
        const CompilerOptions& options);
    ShapeInferenceStats RunShapeInference(
        const std::shared_ptr<Model>& model,
        const CompilerOptions& options);
    PatternMatchingSummary RunPatternMatchingInternal(
        const std::shared_ptr<Model>& model,
        const CompilerOptions& options);
    std::vector<PatternMatchResult> MatchSinglePattern(
        const std::shared_ptr<Model>& model,
        const PatternDefinition& pattern,
        const CompilerOptions& options);
    MatchedNodeInfo ExtractNodeInfo(
        const std::shared_ptr<Node>& node,
        const std::shared_ptr<Model>& model);
    std::string SerializeAttributeValue(const AttributeValue& value);
};

/**
 * @brief Convenience function to compile a model
 */
CompilationResult CompileModel(
    const std::string& input_path,
    const std::string& output_path = "",
    const std::vector<PatternDefinition>& patterns = {},
    const CompilerOptions& options = {});

/**
 * @brief Create common pattern definitions
 */
std::vector<PatternDefinition> GetCommonPatterns();

/**
 * @brief Create pattern from JSON string (for programmatic pattern definition)
 */
std::optional<PatternDefinition> PatternFromJson(const std::string& json_str);

}  // namespace passes
}  // namespace oniris
