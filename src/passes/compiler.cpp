/**
 * @file compiler.cpp
 * @brief ONNX Model Compiler implementation
 */

#include "passes/compiler.hpp"
#include "utils/onnx_utils.hpp"
#include "core/logger.hpp"
#include "ir/tensor.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace oniris {
namespace passes {

// ============================================================================
// Helper functions for JSON serialization
// ============================================================================

static std::string EscapeJsonString(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c >= 0x20 && c <= 0x7E) {
                    oss << c;
                } else {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)(unsigned char)c;
                }
        }
    }
    return oss.str();
}

static std::string ShapeToJsonArray(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        if (shape[i] < 0) {
            oss << "null";  // Dynamic dimension
        } else {
            oss << shape[i];
        }
    }
    oss << "]";
    return oss.str();
}

static std::string ShapeToJsonArray(const std::vector<std::vector<int64_t>>& shapes) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << ShapeToJsonArray(shapes[i]);
    }
    oss << "]";
    return oss.str();
}

template<typename T>
static std::string VectorToJsonArray(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << EscapeJsonString(vec[i]) << "\"";
    }
    oss << "]";
    return oss.str();
}

static std::string MapIntToJsonObject(const std::unordered_map<std::string, int>& map) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first) oss << ", ";
        first = false;
        oss << "\"" << EscapeJsonString(key) << "\": " << value;
    }
    oss << "}";
    return oss.str();
}

static std::string MapStringToJsonObject(const std::unordered_map<std::string, std::string>& map) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first) oss << ", ";
        first = false;
        oss << "\"" << EscapeJsonString(key) << "\": \"" << EscapeJsonString(value) << "\"";
    }
    oss << "}";
    return oss.str();
}

// ============================================================================
// PatternDefinition implementation
// ============================================================================

bool PatternDefinition::Parse(std::string* error_msg) {
    auto parsed = OnnxMatcherPattern::FromString(pattern_string);
    if (!parsed) {
        if (error_msg) {
            *error_msg = "Failed to parse pattern: " + pattern_string;
        }
        return false;
    }
    parsed_pattern = *parsed;
    return true;
}

// ============================================================================
// CompilationResult implementation
// ============================================================================

std::string CompilationResult::ToJson(bool pretty) const {
    std::ostringstream oss;
    const std::string indent = pretty ? "  " : "";
    const std::string newline = pretty ? "\n" : "";
    
    auto inc_indent = [&](int level) -> std::string {
        return pretty ? std::string(level * 2, ' ') : "";
    };
    
    oss << "{" << newline;
    
    // Basic info
    oss << inc_indent(1) << "\"success\": " << (success ? "true" : "false") << "," << newline;
    if (!error_msg.empty()) {
        oss << inc_indent(1) << "\"error_msg\": \"" << EscapeJsonString(error_msg) << "\"," << newline;
    }
    oss << inc_indent(1) << "\"input_path\": \"" << EscapeJsonString(input_path) << "\"," << newline;
    oss << inc_indent(1) << "\"output_path\": \"" << EscapeJsonString(output_path) << "\"," << newline;
    
    // Timing
    oss << inc_indent(1) << "\"timing\": {" << newline;
    oss << inc_indent(2) << "\"start_time\": \"" << EscapeJsonString(start_time) << "\"," << newline;
    oss << inc_indent(2) << "\"end_time\": \"" << EscapeJsonString(end_time) << "\"," << newline;
    oss << inc_indent(2) << "\"duration_ms\": " << std::fixed << std::setprecision(2) << duration_ms << newline;
    oss << inc_indent(1) << "}," << newline;
    
    // Model summary
    oss << inc_indent(1) << "\"model_info\": {" << newline;
    oss << inc_indent(2) << "\"producer_name\": \"" << EscapeJsonString(model_info.producer_name) << "\"," << newline;
    oss << inc_indent(2) << "\"producer_version\": \"" << EscapeJsonString(model_info.producer_version) << "\"," << newline;
    oss << inc_indent(2) << "\"ir_version\": " << model_info.ir_version << "," << newline;
    oss << inc_indent(2) << "\"opset_version\": " << model_info.opset_version << "," << newline;
    oss << inc_indent(2) << "\"num_nodes\": " << model_info.num_nodes << "," << newline;
    oss << inc_indent(2) << "\"num_initializers\": " << model_info.num_initializers << "," << newline;
    oss << inc_indent(2) << "\"num_inputs\": " << model_info.num_inputs << "," << newline;
    oss << inc_indent(2) << "\"num_outputs\": " << model_info.num_outputs << "," << newline;
    oss << inc_indent(2) << "\"op_types_used\": " << VectorToJsonArray(model_info.op_types_used) << "," << newline;
    oss << inc_indent(2) << "\"op_type_counts\": " << MapIntToJsonObject(model_info.op_type_counts) << newline;
    oss << inc_indent(1) << "}," << newline;
    
    // Optimization stats
    oss << inc_indent(1) << "\"optimization\": {" << newline;
    oss << inc_indent(2) << "\"success\": " << (optimization_stats.success ? "true" : "false") << "," << newline;
    if (!optimization_stats.error_msg.empty()) {
        oss << inc_indent(2) << "\"error_msg\": \"" << EscapeJsonString(optimization_stats.error_msg) << "\"," << newline;
    }
    oss << inc_indent(2) << "\"num_iterations\": " << optimization_stats.num_iterations << "," << newline;
    oss << inc_indent(2) << "\"num_changes\": " << optimization_stats.num_changes << "," << newline;
    oss << inc_indent(2) << "\"pass_stats\": {" << newline;
    {
        bool first = true;
        for (const auto& [pass, count] : optimization_stats.pass_stats) {
            if (!first) oss << "," << newline;
            first = false;
            oss << inc_indent(3) << "\"" << EscapeJsonString(pass) << "\": " << count;
        }
        if (!optimization_stats.pass_stats.empty()) oss << newline;
    }
    oss << inc_indent(2) << "}," << newline;
    oss << inc_indent(2) << "\"unsupported_ops\": " << VectorToJsonArray(optimization_stats.unsupported_ops) << newline;
    oss << inc_indent(1) << "}," << newline;
    
    // Shape inference stats
    oss << inc_indent(1) << "\"shape_inference\": {" << newline;
    oss << inc_indent(2) << "\"success\": " << (shape_inference_stats.success ? "true" : "false") << "," << newline;
    if (!shape_inference_stats.error_msg.empty()) {
        oss << inc_indent(2) << "\"error_msg\": \"" << EscapeJsonString(shape_inference_stats.error_msg) << "\"," << newline;
    }
    oss << inc_indent(2) << "\"num_nodes_processed\": " << shape_inference_stats.num_nodes_processed << "," << newline;
    oss << inc_indent(2) << "\"num_nodes_failed\": " << shape_inference_stats.num_nodes_failed << "," << newline;
    oss << inc_indent(2) << "\"failed_nodes\": " << VectorToJsonArray(shape_inference_stats.failed_nodes) << newline;
    oss << inc_indent(1) << "}," << newline;
    
    // Pattern matching results
    oss << inc_indent(1) << "\"pattern_matching\": {" << newline;
    oss << inc_indent(2) << "\"total_patterns\": " << pattern_matching_summary.total_patterns << "," << newline;
    oss << inc_indent(2) << "\"patterns_with_matches\": " << pattern_matching_summary.patterns_with_matches << "," << newline;
    oss << inc_indent(2) << "\"total_matches\": " << pattern_matching_summary.total_matches << "," << newline;
    oss << inc_indent(2) << "\"match_counts\": {" << newline;
    {
        bool first = true;
        for (const auto& [pattern, count] : pattern_matching_summary.match_counts) {
            if (!first) oss << "," << newline;
            first = false;
            oss << inc_indent(3) << "\"" << EscapeJsonString(pattern) << "\": " << count;
        }
        if (!pattern_matching_summary.match_counts.empty()) oss << newline;
    }
    oss << inc_indent(2) << "}," << newline;
    oss << inc_indent(2) << "\"results\": {" << newline;
    
    // Detailed pattern match results
    {
        bool first_pattern = true;
        for (const auto& [pattern_name, matches] : pattern_matching_summary.pattern_results) {
            if (!first_pattern) oss << "," << newline;
            first_pattern = false;
            oss << inc_indent(3) << "\"" << EscapeJsonString(pattern_name) << "\": [" << newline;
            
            for (size_t i = 0; i < matches.size(); ++i) {
                const auto& match = matches[i];
                if (i > 0) oss << "," << newline;
                oss << inc_indent(4) << "{" << newline;
                oss << inc_indent(5) << "\"match_id\": " << match.match_id << "," << newline;
                oss << inc_indent(5) << "\"tensor_bindings\": " << MapStringToJsonObject(match.tensor_bindings) << "," << newline;
                oss << inc_indent(5) << "\"nodes\": [" << newline;
                
                for (size_t j = 0; j < match.nodes.size(); ++j) {
                    const auto& node = match.nodes[j];
                    if (j > 0) oss << "," << newline;
                    oss << inc_indent(6) << "{" << newline;
                    oss << inc_indent(7) << "\"name\": \"" << EscapeJsonString(node.node_name) << "\"," << newline;
                    oss << inc_indent(7) << "\"op_type\": \"" << EscapeJsonString(node.op_type) << "\"," << newline;
                    oss << inc_indent(7) << "\"domain\": \"" << EscapeJsonString(node.domain) << "\"," << newline;
                    oss << inc_indent(7) << "\"inputs\": " << VectorToJsonArray(node.inputs) << "," << newline;
                    oss << inc_indent(7) << "\"outputs\": " << VectorToJsonArray(node.outputs) << "," << newline;
                    oss << inc_indent(7) << "\"input_shapes\": " << ShapeToJsonArray(node.input_shapes) << "," << newline;
                    oss << inc_indent(7) << "\"output_shapes\": " << ShapeToJsonArray(node.output_shapes) << newline;
                    oss << inc_indent(6) << "}";
                }
                
                oss << newline << inc_indent(5) << "]" << newline;
                oss << inc_indent(4) << "}";
            }
            
            if (!matches.empty()) oss << newline;
            oss << inc_indent(3) << "]";
        }
        if (!pattern_matching_summary.pattern_results.empty()) oss << newline;
    }
    
    oss << inc_indent(2) << "}" << newline;
    oss << inc_indent(1) << "}" << newline;
    oss << "}";
    
    return oss.str();
}

bool CompilationResult::SaveJson(const std::string& filepath, bool pretty) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    file << ToJson(pretty);
    return file.good();
}

// ============================================================================
// ModelCompiler implementation
// ============================================================================

ModelCompiler::ModelCompiler() = default;

bool ModelCompiler::AddPattern(const PatternDefinition& pattern) {
    PatternDefinition copy = pattern;
    std::string error_msg;
    if (!copy.Parse(&error_msg)) {
        ONIRIS_WARNING << "Failed to add pattern '" << pattern.name << "': " << error_msg;
        return false;
    }
    patterns_.push_back(std::move(copy));
    return true;
}

bool ModelCompiler::AddPattern(const std::string& name, const std::string& pattern_string) {
    return AddPattern(PatternDefinition(name, pattern_string));
}

void ModelCompiler::AddPatterns(const std::vector<PatternDefinition>& patterns) {
    for (const auto& p : patterns) {
        AddPattern(p);
    }
}

void ModelCompiler::ClearPatterns() {
    patterns_.clear();
}

std::vector<std::string> ModelCompiler::GetPatternNames() const {
    std::vector<std::string> names;
    names.reserve(patterns_.size());
    for (const auto& p : patterns_) {
        names.push_back(p.name);
    }
    return names;
}

void ModelCompiler::Log(const std::string& msg, const CompilerOptions& options) {
    if (options.verbose) {
        ONIRIS_INFO << msg;
    }
    if (options.log_callback) {
        options.log_callback(msg);
    }
}

ModelSummary ModelCompiler::ExtractModelInfo(const std::shared_ptr<Model>& model) {
    ModelSummary summary;
    summary.producer_name = model->GetProducerName();
    summary.producer_version = model->GetProducerVersion();
    summary.ir_version = model->GetIRVersion();
    
    if (!model->GetOpsetImports().empty()) {
        summary.opset_version = model->GetOpsetImports()[0].version;
    }
    
    auto graph = model->GetGraph();
    if (!graph) return summary;
    
    summary.num_nodes = static_cast<int>(graph->GetNodes().size());
    summary.num_inputs = static_cast<int>(graph->GetInputs().size());
    summary.num_outputs = static_cast<int>(graph->GetOutputs().size());
    summary.num_initializers = static_cast<int>(graph->GetInitializers().size());
    
    // Count op types
    for (const auto& node : graph->GetNodes()) {
        std::string op_type = node->GetOpType();
        summary.op_type_counts[op_type]++;
    }
    
    // Get unique op types
    for (const auto& [op_type, _] : summary.op_type_counts) {
        summary.op_types_used.push_back(op_type);
    }
    std::sort(summary.op_types_used.begin(), summary.op_types_used.end());
    
    return summary;
}

OptimizationStats ModelCompiler::RunOptimization(
    const std::shared_ptr<Model>& model,
    const CompilerOptions& options) {
    
    OptimizationStats stats;
    
    if (!options.enable_optimization) {
        Log("Optimization disabled, skipping...", options);
        return stats;
    }
    
    Log("Running optimization passes...", options);
    auto result = Simplifier::Simplify(model, options.simplify_options);
    
    stats.success = result.success;
    stats.error_msg = result.error_msg;
    stats.num_iterations = result.num_iterations;
    stats.num_changes = result.num_changes;
    stats.pass_stats = result.pass_stats;
    stats.unsupported_ops = result.unsupported_ops;
    
    Log("Optimization completed: " + std::to_string(result.num_changes) + " changes in " + 
        std::to_string(result.num_iterations) + " iterations", options);
    
    return stats;
}

ShapeInferenceStats ModelCompiler::RunShapeInference(
    const std::shared_ptr<Model>& model,
    const CompilerOptions& options) {
    
    ShapeInferenceStats stats;
    
    if (!options.enable_shape_inference) {
        Log("Shape inference disabled, skipping...", options);
        return stats;
    }
    
    Log("Running shape inference...", options);
    
    auto& engine = ShapeInferenceEngine::GetInstance();
    auto graph = model->GetGraph();
    if (!graph) {
        stats.success = false;
        stats.error_msg = "Model has no graph";
        return stats;
    }
    
    int total_nodes = static_cast<int>(graph->GetNodes().size());
    int processed = 0;
    int failed = 0;
    
    for (const auto& node : graph->GetNodes()) {
        auto result = engine.InferNode(node, *graph);
        if (result.success) {
            processed++;
        } else {
            failed++;
            stats.failed_nodes.push_back(node->GetName());
        }
    }
    
    stats.num_nodes_processed = processed;
    stats.num_nodes_failed = failed;
    stats.success = (failed == 0) || !options.fail_on_unknown_shape;
    
    Log("Shape inference completed: " + std::to_string(processed) + "/" + 
        std::to_string(total_nodes) + " nodes processed", options);
    
    return stats;
}

MatchedNodeInfo ModelCompiler::ExtractNodeInfo(
    const std::shared_ptr<Node>& node,
    const std::shared_ptr<Model>& model) {
    
    MatchedNodeInfo info;
    info.node_name = node->GetName();
    info.op_type = node->GetOpType();
    info.domain = node->GetDomain();
    info.inputs = node->GetInputs();
    info.outputs = node->GetOutputs();
    
    // Extract shapes from graph value info
    auto graph = model->GetGraph();
    if (graph) {
        // Input shapes
        for (const auto& input_name : node->GetInputs()) {
            if (auto value_info = graph->GetValueInfo(input_name)) {
                std::vector<int64_t> shape;
                for (size_t i = 0; i < value_info->shape.NumDims(); ++i) {
                    const auto& dim = value_info->shape.GetDim(i);
                    if (dim.IsDynamic()) {
                        shape.push_back(-1);
                    } else {
                        shape.push_back(dim.GetStaticValue());
                    }
                }
                info.input_shapes.push_back(std::move(shape));
            } else {
                info.input_shapes.push_back({});  // Unknown shape
            }
        }
        
        // Output shapes
        for (const auto& output_name : node->GetOutputs()) {
            if (auto value_info = graph->GetValueInfo(output_name)) {
                std::vector<int64_t> shape;
                for (size_t i = 0; i < value_info->shape.NumDims(); ++i) {
                    const auto& dim = value_info->shape.GetDim(i);
                    if (dim.IsDynamic()) {
                        shape.push_back(-1);
                    } else {
                        shape.push_back(dim.GetStaticValue());
                    }
                }
                info.output_shapes.push_back(std::move(shape));
            } else {
                info.output_shapes.push_back({});  // Unknown shape
            }
        }
    }
    
    return info;
}

std::vector<PatternMatchResult> ModelCompiler::MatchSinglePattern(
    const std::shared_ptr<Model>& model,
    const PatternDefinition& pattern,
    const CompilerOptions& options) {
    
    std::vector<PatternMatchResult> results;
    
    if (!pattern.parsed_pattern) {
        return results;
    }
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern.parsed_pattern);
    
    int match_id = 0;
    for (const auto& match : matches) {
        if (!match.IsValid()) continue;
        
        PatternMatchResult result;
        result.pattern_name = pattern.name;
        result.match_id = match_id++;
        
        // Extract node info
        for (size_t i = 0; i < match.matched_nodes.size(); ++i) {
            const auto& node = match.matched_nodes[i];
            auto node_info = ExtractNodeInfo(node, model);
            result.nodes.push_back(node_info);
            result.node_mapping["node" + std::to_string(i)] = node_info;
        }
        
        results.push_back(std::move(result));
        
        // Respect max matches limit
        if (match_id >= options.max_matches_per_pattern) {
            break;
        }
    }
    
    return results;
}

PatternMatchingSummary ModelCompiler::RunPatternMatchingInternal(
    const std::shared_ptr<Model>& model,
    const CompilerOptions& options) {
    
    PatternMatchingSummary summary;
    summary.total_patterns = static_cast<int>(patterns_.size());
    
    if (!options.enable_pattern_matching || patterns_.empty()) {
        Log("Pattern matching disabled or no patterns registered, skipping...", options);
        return summary;
    }
    
    Log("Running pattern matching with " + std::to_string(patterns_.size()) + " patterns...", options);
    
    for (const auto& pattern : patterns_) {
        auto results = MatchSinglePattern(model, pattern, options);
        
        if (!results.empty()) {
            summary.patterns_with_matches++;
            summary.total_matches += static_cast<int>(results.size());
            summary.pattern_results[pattern.name] = std::move(results);
            summary.match_counts[pattern.name] = static_cast<int>(summary.pattern_results[pattern.name].size());
        } else {
            summary.match_counts[pattern.name] = 0;
        }
    }
    
    Log("Pattern matching completed: " + std::to_string(summary.total_matches) + 
        " matches found across " + std::to_string(summary.patterns_with_matches) + " patterns", options);
    
    return summary;
}

CompilationResult ModelCompiler::Compile(
    const std::string& input_path,
    const std::string& output_path,
    const CompilerOptions& options) {
    
    CompilationResult result;
    result.input_path = input_path;
    result.output_path = output_path;
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss;
    time_ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    result.start_time = time_ss.str();
    
    Log("Starting compilation of: " + input_path, options);
    
    // Load model
    Log("Loading model...", options);
    auto model = utils::LoadModel(input_path);
    if (!model) {
        result.success = false;
        result.error_msg = "Failed to load model: " + input_path;
        ONIRIS_ERROR << result.error_msg;
        return result;
    }
    
    // Extract model info
    result.model_info = ExtractModelInfo(model);
    Log("Model loaded: " + std::to_string(result.model_info.num_nodes) + " nodes, " +
        std::to_string(result.model_info.num_inputs) + " inputs, " +
        std::to_string(result.model_info.num_outputs) + " outputs", options);
    
    // Run optimization
    result.optimization_stats = RunOptimization(model, options);
    if (!result.optimization_stats.success && options.simplify_options.fail_on_unsupported) {
        result.success = false;
        result.error_msg = "Optimization failed: " + result.optimization_stats.error_msg;
        return result;
    }
    
    // Run shape inference
    result.shape_inference_stats = RunShapeInference(model, options);
    if (!result.shape_inference_stats.success) {
        result.success = false;
        result.error_msg = "Shape inference failed: " + result.shape_inference_stats.error_msg;
        return result;
    }
    
    // Run pattern matching
    result.pattern_matching_summary = RunPatternMatchingInternal(model, options);
    
    // Save optimized model if requested
    if (options.save_optimized_model && !output_path.empty()) {
        Log("Saving optimized model to: " + output_path, options);
        if (!utils::SaveModel(model, output_path)) {
            result.success = false;
            result.error_msg = "Failed to save model: " + output_path;
            return result;
        }
    }
    
    // Save JSON result if requested
    if (options.save_json_result && !options.json_output_path.empty()) {
        Log("Saving JSON result to: " + options.json_output_path, options);
        if (!result.SaveJson(options.json_output_path)) {
            ONIRIS_WARNING << "Failed to save JSON result to: " << options.json_output_path;
        }
    }
    
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_time_sys = std::chrono::system_clock::now();
    auto time_t_end = std::chrono::system_clock::to_time_t(end_time_sys);
    std::stringstream time_ss_end;
    time_ss_end << std::put_time(std::localtime(&time_t_end), "%Y-%m-%d %H:%M:%S");
    result.end_time = time_ss_end.str();
    
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    result.duration_ms = duration.count();
    
    Log("Compilation completed in " + std::to_string(result.duration_ms) + " ms", options);
    
    return result;
}

CompilationResult ModelCompiler::CompileModel(
    const std::shared_ptr<Model>& model,
    const CompilerOptions& options) {
    
    CompilationResult result;
    
    if (!model) {
        result.success = false;
        result.error_msg = "Null model pointer";
        return result;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract model info
    result.model_info = ExtractModelInfo(model);
    
    // Optionally run pattern matching before optimization
    PatternMatchingSummary pre_opt_summary;
    if (options.enable_pattern_matching && options.pattern_match_before_opt) {
        Log("Running pattern matching before optimization...", options);
        pre_opt_summary = RunPatternMatchingInternal(model, options);
    }
    
    // Run optimization
    result.optimization_stats = RunOptimization(model, options);
    
    // Run shape inference
    result.shape_inference_stats = RunShapeInference(model, options);
    
    // Run pattern matching (after optimization, or use pre-optimization results)
    if (options.enable_pattern_matching) {
        if (options.pattern_match_before_opt) {
            // Use the pre-optimization results
            result.pattern_matching_summary = pre_opt_summary;
        } else {
            // Run pattern matching on optimized model
            result.pattern_matching_summary = RunPatternMatchingInternal(model, options);
        }
    }
    
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    result.duration_ms = duration.count();
    
    return result;
}

PatternMatchingSummary ModelCompiler::RunPatternMatching(
    const std::shared_ptr<Model>& model,
    const PatternMatchType& match_type) {
    
    CompilerOptions options;
    options.enable_pattern_matching = true;
    options.match_type = match_type;
    return RunPatternMatchingInternal(model, options);
}

// ============================================================================
// Convenience functions
// ============================================================================

CompilationResult CompileModel(
    const std::string& input_path,
    const std::string& output_path,
    const std::vector<PatternDefinition>& patterns,
    const CompilerOptions& options) {
    
    ModelCompiler compiler;
    compiler.AddPatterns(patterns);
    return compiler.Compile(input_path, output_path, options);
}

std::vector<PatternDefinition> GetCommonPatterns() {
    std::vector<PatternDefinition> patterns;
    
    // Conv + ReLU fusion pattern
    patterns.emplace_back("ConvRelu", R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    // Conv + BatchNorm + ReLU pattern
    patterns.emplace_back("ConvBnRelu", R"(
        Conv(?, c0)
        BatchNormalization(c0, bn0)
        Relu(bn0, ?)
    )");
    
    // Swish activation: x * sigmoid(x)
    patterns.emplace_back("Swish", R"(
        Conv(?, c0)
        Sigmoid(c0, s0)
        Mul([s0, c0], ?)
    )");
    
    // GELU activation pattern
    patterns.emplace_back("Gelu", R"(
        Div(?, div0)
        Erf(div0, erf0)
        Add(?, add0)
        Mul(?, mul0)
        Mul([mul0, ?], ?)
    )");
    
    // Residual connection: Conv + Add
    patterns.emplace_back("Residual", R"(
        Conv(?, c0)
        Add([c0, ?], ?)
    )");
    
    // Multi-head attention pattern (simplified)
    patterns.emplace_back("Attention", R"(
        MatMul(?, qk0)
        Softmax(qk0, softmax0)
        MatMul([softmax0, ?], ?)
    )");
    
    // LayerNorm pattern
    patterns.emplace_back("LayerNorm", R"(
        ReduceMean(?, rm0)
        Sub([?, rm0], sub0)
        Pow(sub0, pow0)
        ReduceMean(pow0, rm1)
        Add(rm1, add0)
        Sqrt(add0, sqrt0)
        Div(sub0, div0)
    )");
    
    return patterns;
}

}  // namespace passes
}  // namespace oniris
