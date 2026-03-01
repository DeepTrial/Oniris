/**
 * @file bindings.cpp
 * @brief Python bindings for Oniris using pybind11
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "core/types.hpp"
#include "ir/graph.hpp"
#include "ir/model.hpp"
#include "ir/node.hpp"
#include "ir/tensor.hpp"
#include "passes/shape_inference.hpp"
#include "passes/simplifier.hpp"
#include "passes/onnx_matcher_style.hpp"
#include "passes/compiler.hpp"
#include "passes/pattern_manager.hpp"
#include "utils/onnx_utils.hpp"

namespace py = pybind11;

using namespace oniris;
using namespace oniris::passes;
using namespace oniris::utils;

// Helper to convert Python list to Shape
Shape ListToShape(const py::list& dims) {
    std::vector<Dimension> dimensions;
    for (auto item : dims) {
        if (py::isinstance<py::int_>(item)) {
            dimensions.emplace_back(item.cast<int64_t>());
        } else if (py::isinstance<py::str>(item)) {
            dimensions.emplace_back(item.cast<std::string>());
        } else if (py::isinstance<Dimension>(item)) {
            dimensions.emplace_back(item.cast<Dimension>());
        } else if (item.is_none()) {
            dimensions.emplace_back();  // Dynamic
        }
    }
    return Shape(dimensions);
}

// Helper to convert Shape to Python list
py::list ShapeToList(const Shape& shape) {
    py::list result;
    for (size_t i = 0; i < shape.NumDims(); ++i) {
        const auto& dim = shape.GetDim(i);
        if (dim.IsDynamic()) {
            if (dim.GetSymbolicName().empty()) {
                result.append(py::none());
            } else {
                result.append(dim.GetSymbolicName());
            }
        } else {
            result.append(dim.GetStaticValue());
        }
    }
    return result;
}

PYBIND11_MODULE(_oniris, m) {
    m.doc() = "Oniris - ONNX Compilation Toolkit";
    
    // ========================================================================
    // Core Types
    // ========================================================================
    
    py::enum_<DataType>(m, "DataType")
        .value("UNDEFINED", DataType::kUndefined)
        .value("FLOAT", DataType::kFloat)
        .value("UINT8", DataType::kUint8)
        .value("INT8", DataType::kInt8)
        .value("UINT16", DataType::kUint16)
        .value("INT16", DataType::kInt16)
        .value("INT32", DataType::kInt32)
        .value("INT64", DataType::kInt64)
        .value("STRING", DataType::kString)
        .value("BOOL", DataType::kBool)
        .value("FLOAT16", DataType::kFloat16)
        .value("DOUBLE", DataType::kDouble)
        .value("UINT32", DataType::kUint32)
        .value("UINT64", DataType::kUint64)
        .value("COMPLEX64", DataType::kComplex64)
        .value("COMPLEX128", DataType::kComplex128)
        .value("BFLOAT16", DataType::kBFloat16)
        // FP8 types
        .value("FLOAT8E4M3FN", DataType::kFloat8E4M3FN)
        .value("FLOAT8E4M3FNUZ", DataType::kFloat8E4M3FNUZ)
        .value("FLOAT8E5M2", DataType::kFloat8E5M2)
        .value("FLOAT8E5M2FNUZ", DataType::kFloat8E5M2FNUZ)
        // 4-bit types
        .value("UINT4", DataType::kUint4)
        .value("INT4", DataType::kInt4)
        // .value("FLOAT4E2M1", DataType::kFloat4E2M1)  // ONNX 1.17+
        // Aliases for backward compatibility
        .value("UNKNOWN", DataType::kUnknown)
        .value("FLOAT32", DataType::kFloat32)
        .value("FLOAT64", DataType::kFloat64)
        .export_values();
    
    m.def("data_type_to_string", &DataTypeToString, "Convert DataType to string");
    m.def("string_to_data_type", &StringToDataType, "Convert string to DataType");
    
    // Dimension
    py::class_<Dimension>(m, "Dimension")
        .def(py::init<int64_t>(), "Create static dimension")
        .def(py::init<std::string>(), "Create dynamic dimension with symbol")
        .def("is_dynamic", &Dimension::IsDynamic, "Check if dimension is dynamic")
        .def("get_static_value", &Dimension::GetStaticValue, "Get static value")
        .def("get_symbolic_name", &Dimension::GetSymbolicName, "Get symbolic name")
        .def("set_static_value", &Dimension::SetStaticValue, "Set static value")
        .def("set_dynamic", &Dimension::SetDynamic, "Set as dynamic")
        .def("__str__", &Dimension::ToString)
        .def("__repr__", [](const Dimension& d) {
            return "Dimension(" + d.ToString() + ")";
        });
    
    // Shape
    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<std::vector<Dimension>>())
        .def(py::init([](const py::list& dims) {
            return ListToShape(dims);
        }))
        .def("num_dims", &Shape::NumDims, "Get number of dimensions")
        .def("is_scalar", &Shape::IsScalar, "Check if scalar")
        .def("is_dynamic", &Shape::IsDynamic, "Check if any dimension is dynamic")
        .def("is_static", &Shape::IsStatic, "Check if all dimensions are static")
        .def("get_dim", [](Shape& s, size_t idx) -> Dimension& { return s.GetDim(idx); }, 
             py::arg("idx"), py::return_value_policy::reference)
        .def("get_dims", static_cast<const std::vector<Dimension>& (Shape::*)() const>(&Shape::GetDims))
        .def("add_dim", py::overload_cast<const Dimension&>(&Shape::AddDim))
        .def("add_dim", py::overload_cast<int64_t>(&Shape::AddDim))
        .def("get_total_size", &Shape::GetTotalSize)
        .def("to_list", &ShapeToList)
        .def("__str__", &Shape::ToString)
        .def("__repr__", [](const Shape& s) {
            return "Shape(" + s.ToString() + ")";
        });
    
    // ========================================================================
    // IR Classes
    // ========================================================================
    
    // Node
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<std::string, std::string>(), 
             py::arg("op_type"), py::arg("name") = "")
        .def("get_op_type", &Node::GetOpType)
        .def("set_op_type", &Node::SetOpType)
        .def("get_name", &Node::GetName)
        .def("set_name", &Node::SetName)
        .def("get_domain", &Node::GetDomain)
        .def("set_domain", &Node::SetDomain)
        .def("get_inputs", py::overload_cast<>(&Node::GetInputs), py::return_value_policy::reference)
        .def("add_input", &Node::AddInput)
        .def("get_outputs", py::overload_cast<>(&Node::GetOutputs), py::return_value_policy::reference)
        .def("add_output", &Node::AddOutput)
        .def("has_attribute", &Node::HasAttribute)
        .def("get_attribute", [](const Node& n, const std::string& name) -> py::object {
            auto* attr = n.GetAttribute(name);
            if (!attr) return py::none();
            
            return std::visit([](auto&& arg) -> py::object {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, int64_t>) {
                    return py::int_(arg);
                } else if constexpr (std::is_same_v<T, float>) {
                    return py::float_(arg);
                } else if constexpr (std::is_same_v<T, std::string>) {
                    return py::str(arg);
                } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    return py::cast(arg);
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    return py::cast(arg);
                } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                    return py::cast(arg);
                } else {
                    return py::none();
                }
            }, *attr);
        })
        .def("set_attribute_int", [](Node& n, const std::string& name, int64_t value) {
            n.SetAttribute(name, value);
        })
        .def("set_attribute_float", [](Node& n, const std::string& name, float value) {
            n.SetAttribute(name, value);
        })
        .def("set_attribute_string", [](Node& n, const std::string& name, const std::string& value) {
            n.SetAttribute(name, value);
        })
        .def("set_attribute_ints", [](Node& n, const std::string& name, const std::vector<int64_t>& value) {
            n.SetAttribute(name, value);
        })
        .def("set_attribute_floats", [](Node& n, const std::string& name, const std::vector<float>& value) {
            n.SetAttribute(name, value);
        })
        .def("remove_attribute", &Node::RemoveAttribute)
        .def("has_inferred_shapes", &Node::HasInferredShapes);
    
    // ValueInfo
    py::class_<ValueInfo>(m, "ValueInfo")
        .def(py::init<>())
        .def_readwrite("name", &ValueInfo::name)
        .def_readwrite("shape", &ValueInfo::shape)
        .def_readwrite("dtype", &ValueInfo::dtype)
        .def("has_inferred_shape", &ValueInfo::HasInferredShape);
    
    // Graph
    py::class_<Graph, std::shared_ptr<Graph>>(m, "Graph")
        .def(py::init<std::string>(), py::arg("name") = "")
        .def("get_name", &Graph::GetName)
        .def("set_name", &Graph::SetName)
        .def("get_nodes", py::overload_cast<>(&Graph::GetNodes), py::return_value_policy::reference)
        .def("add_node", &Graph::AddNode)
        .def("create_node", &Graph::CreateNode,
             py::arg("op_type"), py::arg("name") = "",
             py::return_value_policy::reference)
        .def("remove_node", &Graph::RemoveNode)
        .def("remove_dead_nodes", &Graph::RemoveDeadNodes)
        .def("get_inputs", py::overload_cast<>(&Graph::GetInputs), py::return_value_policy::reference)
        .def("add_input", &Graph::AddInput)
        .def("set_inputs", &Graph::SetInputs)
        .def("get_outputs", py::overload_cast<>(&Graph::GetOutputs), py::return_value_policy::reference)
        .def("add_output", &Graph::AddOutput)
        .def("set_outputs", &Graph::SetOutputs)
        .def("has_value_info", &Graph::HasValueInfo)
        .def("get_value_info", [](Graph& g, const std::string& name) -> ValueInfo* {
            return g.GetValueInfo(name);
        }, py::return_value_policy::reference)
        .def("set_value_info", &Graph::SetValueInfo)
        .def("get_value_info_names", [](const Graph& g) {
            std::vector<std::string> names;
            for (const auto& [name, _] : g.GetAllValueInfos()) {
                names.push_back(name);
            }
            return names;
        })
        .def("get_producer", &Graph::GetProducer)
        .def("get_consumers", &Graph::GetConsumers)
        .def("topological_sort", &Graph::TopologicalSort)
        .def("validate", [](const Graph& g) {
            std::string msg;
            bool ok = g.Validate(&msg);
            return py::make_tuple(ok, msg);
        });
    
    // OpsetImport
    py::class_<OpsetImport>(m, "OpsetImport")
        .def(py::init<>())
        .def_readwrite("domain", &OpsetImport::domain)
        .def_readwrite("version", &OpsetImport::version);
    
    // Model
    py::class_<Model, std::shared_ptr<Model>>(m, "Model")
        .def(py::init<int64_t>(), py::arg("ir_version") = 8)
        .def("get_ir_version", &Model::GetIRVersion)
        .def("set_ir_version", &Model::SetIRVersion)
        .def("get_producer_name", &Model::GetProducerName)
        .def("set_producer_name", &Model::SetProducerName)
        .def("get_producer_version", &Model::GetProducerVersion)
        .def("set_producer_version", &Model::SetProducerVersion)
        .def("get_domain", &Model::GetDomain)
        .def("set_domain", &Model::SetDomain)
        .def("get_model_version", &Model::GetModelVersion)
        .def("set_model_version", &Model::SetModelVersion)
        .def("get_doc_string", &Model::GetDocString)
        .def("set_doc_string", &Model::SetDocString)
        .def("get_opset_imports", py::overload_cast<>(&Model::GetOpsetImports), 
             py::return_value_policy::reference)
        .def("add_opset_import", &Model::AddOpsetImport)
        .def("get_graph", &Model::GetGraph)
        .def("set_graph", &Model::SetGraph)
        .def("create_graph", &Model::CreateGraph, py::return_value_policy::reference)
        .def("validate", [](const Model& m) {
            std::string msg;
            bool ok = m.Validate(&msg);
            return py::make_tuple(ok, msg);
        });
    
    // ========================================================================
    // Passes
    // ========================================================================
    
    // InferenceContext
    py::class_<InferenceContext>(m, "InferenceContext")
        .def_readonly("input_shapes", &InferenceContext::input_shapes)
        .def_readonly("input_dtypes", &InferenceContext::input_dtypes)
        .def("get_input_shape", &InferenceContext::GetInputShape, py::arg("idx"));
    
    // InferenceResult
    py::class_<InferenceResult>(m, "InferenceResult")
        .def_readonly("success", &InferenceResult::success)
        .def_readonly("error_msg", &InferenceResult::error_msg)
        .def_readonly("output_shapes", &InferenceResult::output_shapes)
        .def_static("Success", &InferenceResult::Success, py::arg("shapes"))
        .def_static("Error", &InferenceResult::Error, py::arg("msg"));
    
    // ShapeInferenceEngine
    py::class_<ShapeInferenceEngine>(m, "ShapeInferenceEngine")
        .def_static("get_instance", &ShapeInferenceEngine::GetInstance,
                    py::return_value_policy::reference)
        .def("register_handler", [](ShapeInferenceEngine& self, 
                                    const std::string& op_type,
                                    py::function func,
                                    const std::string& domain) {
            // Note: Storing Python callbacks in C++ can cause GIL issues on exit.
            // For production use, avoid storing Python functions in C++ singletons.
            self.Register(op_type, [func](const InferenceContext& ctx) -> InferenceResult {
                py::gil_scoped_acquire acquire;
                py::object result = func(ctx);
                return result.cast<InferenceResult>();
            }, domain);
        }, py::arg("op_type"), py::arg("func"), py::arg("domain") = "")
        .def("has_handler", &ShapeInferenceEngine::HasHandler,
             py::arg("op_type"), py::arg("domain") = "")
        .def("infer_graph", &ShapeInferenceEngine::InferGraph,
             py::arg("graph"), py::arg("fail_on_unknown") = false)
        .def("infer_model", &ShapeInferenceEngine::InferModel,
             py::arg("model"), py::arg("fail_on_unknown") = false)
        .def("get_supported_ops", &ShapeInferenceEngine::GetSupportedOps);
    
    // SimplifyOptions
    py::class_<SimplifyOptions>(m, "SimplifyOptions")
        .def(py::init<>())
        .def_readwrite("skip_shape_inference", &SimplifyOptions::skip_shape_inference)
        .def_readwrite("skip_constant_folding", &SimplifyOptions::skip_constant_folding)
        .def_readwrite("skip_constant_to_initializer", &SimplifyOptions::skip_constant_to_initializer)
        .def_readwrite("skip_dead_node_elimination", &SimplifyOptions::skip_dead_node_elimination)
        .def_readwrite("skip_identity_elimination", &SimplifyOptions::skip_identity_elimination)
        .def_readwrite("skip_shape_ops_simplification", &SimplifyOptions::skip_shape_ops_simplification)
        .def_readwrite("skip_transpose_elimination", &SimplifyOptions::skip_transpose_elimination)
        .def_readwrite("skip_reshape_elimination", &SimplifyOptions::skip_reshape_elimination)
        .def_readwrite("skip_pad_elimination", &SimplifyOptions::skip_pad_elimination)
        .def_readwrite("skip_slice_elimination", &SimplifyOptions::skip_slice_elimination)
        .def_readwrite("fuse_conv_bn", &SimplifyOptions::fuse_conv_bn)
        .def_readwrite("fuse_conv_relu", &SimplifyOptions::fuse_conv_relu)
        .def_readwrite("fuse_gemm_activation", &SimplifyOptions::fuse_gemm_activation)
        .def_readwrite("fuse_gemm_bias", &SimplifyOptions::fuse_gemm_bias)
        .def_readwrite("fuse_qgemm_activation", &SimplifyOptions::fuse_qgemm_activation)
        .def_readwrite("fail_on_unsupported", &SimplifyOptions::fail_on_unsupported)
        .def_readwrite("max_iterations", &SimplifyOptions::max_iterations)
        .def_readwrite("verbose", &SimplifyOptions::verbose);
    
    // SimplifyResult
    py::class_<SimplifyResult>(m, "SimplifyResult")
        .def_readonly("success", &SimplifyResult::success)
        .def_readonly("error_msg", &SimplifyResult::error_msg)
        .def_readonly("num_changes", &SimplifyResult::num_changes)
        .def_readonly("num_iterations", &SimplifyResult::num_iterations)
        .def_readonly("unsupported_ops", &SimplifyResult::unsupported_ops)
        .def_readonly("pass_stats", &SimplifyResult::pass_stats);
    
    // Simplifier
    py::class_<Simplifier>(m, "Simplifier")
        .def_static("simplify", &Simplifier::Simplify,
                    py::arg("model"), py::arg("options") = SimplifyOptions())
        .def_static("simplify_graph", &Simplifier::SimplifyGraph,
                    py::arg("graph"), py::arg("options") = SimplifyOptions());
    
    // ========================================================================
    // ========================================================================
    // ONNX Matcher Style (Tensor-flow based patterns)
    // ========================================================================
    
    // OnnxMatcherPattern
    py::class_<OnnxMatcherPattern>(m, "OnnxMatcherPattern")
        .def_static("from_string", [](const std::string& pattern_str) {
            return OnnxMatcherPattern::FromString(pattern_str);
        }, py::arg("pattern_str"),
             R"(Parse pattern in onnx_matcher style.

Syntax:
  OpType(input_tensors, output_tensors)
  OpType1/OpType2(input, output)
  ?(input, output)  # wildcard for any op type

Special:
  ?  - matches any tensor or op type
  [a, b] - list of tensors

Examples:
  # Conv -> Sigmoid -> Mul (Swish activation)
  Conv(?, c0)
  Sigmoid(c0, s0)
  Mul([s0, c0], ?)
  
  # Conv/Pool with any input
  Conv/Pool(?, output)
  
  # Multi-head attention pattern
  MatMul(?, q)
  MatMul(?, k)
  MatMul(?, v)
  MatMul([q, k], qk)
  Softmax(qk, attn)
  MatMul([attn, v], out)
)");
    
    // OnnxMatcherStyleMatcher
    py::class_<OnnxMatcherStyleMatcher>(m, "OnnxMatcherStyleMatcher")
        .def_static("find_all", &OnnxMatcherStyleMatcher::FindAll,
                    py::arg("model"), py::arg("pattern"),
                    "Find all matches using onnx_matcher style pattern")
        .def_static("find_first", &OnnxMatcherStyleMatcher::FindFirst,
                    py::arg("model"), py::arg("pattern"),
                    "Find first match using onnx_matcher style pattern")
        .def_static("has_match", &OnnxMatcherStyleMatcher::HasMatch,
                    py::arg("model"), py::arg("pattern"),
                    "Check if pattern exists");
    
    // Note: Use OnnxMatcherStyleMatcher directly, or alias it in Python
    
    // ========================================================================
    // Model Compiler
    // ========================================================================
    
    // PatternMatchType enum
    py::enum_<PatternMatchType>(m, "PatternMatchType")
        .value("FIRST", PatternMatchType::kFirst, "Stop at first match")
        .value("ALL", PatternMatchType::kAll, "Find all matches")
        .value("COUNT_ONLY", PatternMatchType::kCountOnly, "Just count matches")
        .export_values();
    
    // PatternDefinition
    py::class_<PatternDefinition>(m, "PatternDefinition")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("name"), py::arg("pattern_string"),
             "Create pattern definition with name and pattern string")
        .def_readwrite("name", &PatternDefinition::name)
        .def_readwrite("pattern_string", &PatternDefinition::pattern_string)
        .def("parse", &PatternDefinition::Parse, 
             py::arg("error_msg") = py::none(),
             "Parse the pattern string")
        .def_static("from_string", [](const std::string& name, const std::string& pattern) {
            return PatternDefinition(name, pattern);
        }, py::arg("name"), py::arg("pattern_string"));
    
    // MatchedNodeInfo
    py::class_<MatchedNodeInfo>(m, "MatchedNodeInfo")
        .def_readonly("node_name", &MatchedNodeInfo::node_name)
        .def_readonly("op_type", &MatchedNodeInfo::op_type)
        .def_readonly("domain", &MatchedNodeInfo::domain)
        .def_readonly("inputs", &MatchedNodeInfo::inputs)
        .def_readonly("outputs", &MatchedNodeInfo::outputs)
        .def_readonly("input_shapes", &MatchedNodeInfo::input_shapes)
        .def_readonly("output_shapes", &MatchedNodeInfo::output_shapes);
    
    // PatternMatchResult
    py::class_<PatternMatchResult>(m, "PatternMatchResult")
        .def_readonly("pattern_name", &PatternMatchResult::pattern_name)
        .def_readonly("match_id", &PatternMatchResult::match_id)
        .def_readonly("nodes", &PatternMatchResult::nodes)
        .def_readonly("tensor_bindings", &PatternMatchResult::tensor_bindings)
        .def_readonly("node_mapping", &PatternMatchResult::node_mapping);
    
    // PatternMatchingSummary
    py::class_<PatternMatchingSummary>(m, "PatternMatchingSummary")
        .def_readonly("total_patterns", &PatternMatchingSummary::total_patterns)
        .def_readonly("patterns_with_matches", &PatternMatchingSummary::patterns_with_matches)
        .def_readonly("total_matches", &PatternMatchingSummary::total_matches)
        .def_readonly("match_counts", &PatternMatchingSummary::match_counts)
        .def_readonly("pattern_results", &PatternMatchingSummary::pattern_results);
    
    // OptimizationStats
    py::class_<OptimizationStats>(m, "OptimizationStats")
        .def_readonly("success", &OptimizationStats::success)
        .def_readonly("error_msg", &OptimizationStats::error_msg)
        .def_readonly("num_iterations", &OptimizationStats::num_iterations)
        .def_readonly("num_changes", &OptimizationStats::num_changes)
        .def_readonly("pass_stats", &OptimizationStats::pass_stats)
        .def_readonly("unsupported_ops", &OptimizationStats::unsupported_ops);
    
    // ShapeInferenceStats
    py::class_<ShapeInferenceStats>(m, "ShapeInferenceStats")
        .def_readonly("success", &ShapeInferenceStats::success)
        .def_readonly("error_msg", &ShapeInferenceStats::error_msg)
        .def_readonly("num_nodes_processed", &ShapeInferenceStats::num_nodes_processed)
        .def_readonly("num_nodes_failed", &ShapeInferenceStats::num_nodes_failed)
        .def_readonly("failed_nodes", &ShapeInferenceStats::failed_nodes);
    
    // ModelSummary
    py::class_<ModelSummary>(m, "ModelSummary")
        .def_readonly("producer_name", &ModelSummary::producer_name)
        .def_readonly("producer_version", &ModelSummary::producer_version)
        .def_readonly("ir_version", &ModelSummary::ir_version)
        .def_readonly("opset_version", &ModelSummary::opset_version)
        .def_readonly("num_nodes", &ModelSummary::num_nodes)
        .def_readonly("num_initializers", &ModelSummary::num_initializers)
        .def_readonly("num_inputs", &ModelSummary::num_inputs)
        .def_readonly("num_outputs", &ModelSummary::num_outputs)
        .def_readonly("op_types_used", &ModelSummary::op_types_used)
        .def_readonly("op_type_counts", &ModelSummary::op_type_counts);
    
    // CompilationResult
    py::class_<CompilationResult>(m, "CompilationResult")
        .def_readonly("success", &CompilationResult::success)
        .def_readonly("error_msg", &CompilationResult::error_msg)
        .def_readonly("input_path", &CompilationResult::input_path)
        .def_readonly("output_path", &CompilationResult::output_path)
        .def_readonly("model_info", &CompilationResult::model_info)
        .def_readonly("optimization_stats", &CompilationResult::optimization_stats)
        .def_readonly("shape_inference_stats", &CompilationResult::shape_inference_stats)
        .def_readonly("pattern_matching_summary", &CompilationResult::pattern_matching_summary)
        .def_readonly("start_time", &CompilationResult::start_time)
        .def_readonly("end_time", &CompilationResult::end_time)
        .def_readonly("duration_ms", &CompilationResult::duration_ms)
        .def("to_json", &CompilationResult::ToJson, 
             py::arg("pretty") = true,
             "Serialize result to JSON string")
        .def("save_json", &CompilationResult::SaveJson,
             py::arg("filepath"), py::arg("pretty") = true,
             "Save result to JSON file");
    
    // CompilerOptions
    py::class_<CompilerOptions>(m, "CompilerOptions")
        .def(py::init<>())
        .def_readwrite("simplify_options", &CompilerOptions::simplify_options)
        .def_readwrite("enable_optimization", &CompilerOptions::enable_optimization)
        .def_readwrite("enable_shape_inference", &CompilerOptions::enable_shape_inference)
        .def_readwrite("fail_on_unknown_shape", &CompilerOptions::fail_on_unknown_shape)
        .def_readwrite("enable_pattern_matching", &CompilerOptions::enable_pattern_matching)
        .def_readwrite("match_type", &CompilerOptions::match_type)
        .def_readwrite("max_matches_per_pattern", &CompilerOptions::max_matches_per_pattern)
        .def_readwrite("pattern_match_before_opt", &CompilerOptions::pattern_match_before_opt)
        .def_readwrite("save_optimized_model", &CompilerOptions::save_optimized_model)
        .def_readwrite("save_json_result", &CompilerOptions::save_json_result)
        .def_readwrite("json_output_path", &CompilerOptions::json_output_path)
        .def_readwrite("verbose", &CompilerOptions::verbose);
    
    // ModelCompiler
    py::class_<ModelCompiler>(m, "ModelCompiler")
        .def(py::init<>())
        .def("add_pattern", py::overload_cast<const PatternDefinition&>(&ModelCompiler::AddPattern),
             py::arg("pattern"),
             "Add a pattern to match")
        .def("add_pattern", py::overload_cast<const std::string&, const std::string&>(&ModelCompiler::AddPattern),
             py::arg("name"), py::arg("pattern_string"),
             "Add a pattern by name and string")
        .def("add_patterns", &ModelCompiler::AddPatterns,
             py::arg("patterns"),
             "Add multiple patterns")
        .def("clear_patterns", &ModelCompiler::ClearPatterns,
             "Clear all patterns")
        .def("get_pattern_count", &ModelCompiler::GetPatternCount,
             "Get number of registered patterns")
        .def("get_pattern_names", &ModelCompiler::GetPatternNames,
             "Get registered pattern names")
        .def("compile", &ModelCompiler::Compile,
             py::arg("input_path"),
             py::arg("output_path") = "",
             py::arg("options") = CompilerOptions(),
             "Compile a model file")
        .def("compile_model", &ModelCompiler::CompileModel,
             py::arg("model"),
             py::arg("options") = CompilerOptions(),
             "Compile a model object")
        .def("run_pattern_matching", &ModelCompiler::RunPatternMatching,
             py::arg("model"),
             py::arg("match_type") = PatternMatchType::kAll,
             "Run only pattern matching");
    
    // Convenience functions
    m.def("compile_model", py::overload_cast<const std::string&, const std::string&, 
          const std::vector<PatternDefinition>&, const CompilerOptions&>(
          &CompileModel),
          py::arg("input_path"),
          py::arg("output_path") = "",
          py::arg("patterns") = std::vector<PatternDefinition>{},
          py::arg("options") = CompilerOptions{},
          "Compile a model with patterns and options");
    
    m.def("get_common_patterns", &GetCommonPatterns,
          "Get common pattern definitions for typical fusion patterns");
    
    // ========================================================================
    // Pattern Manager
    // ========================================================================
    
    // PatternCategory enum
    py::enum_<PatternCategory>(m, "PatternCategory")
        .value("FUSION", PatternCategory::kFusion, "Operator fusion patterns")
        .value("OPTIMIZATION", PatternCategory::kOptimization, "Optimization patterns")
        .value("QUANTIZATION", PatternCategory::kQuantization, "Quantization patterns")
        .value("CUSTOM", PatternCategory::kCustom, "User-defined patterns")
        .value("ANALYSIS", PatternCategory::kAnalysis, "Analysis patterns")
        .value("ALL", PatternCategory::kAll, "All categories")
        .export_values();
    
    // PatternMetadata
    py::class_<PatternMetadata>(m, "PatternMetadata")
        .def(py::init<>())
        .def_readwrite("name", &PatternMetadata::name)
        .def_readwrite("description", &PatternMetadata::description)
        .def_readwrite("author", &PatternMetadata::author)
        .def_readwrite("version", &PatternMetadata::version)
        .def_readwrite("category", &PatternMetadata::category)
        .def_readwrite("tags", &PatternMetadata::tags)
        .def_readwrite("attributes", &PatternMetadata::attributes)
        .def_readwrite("created_at", &PatternMetadata::created_at)
        .def_readwrite("modified_at", &PatternMetadata::modified_at)
        .def_readwrite("enabled", &PatternMetadata::enabled)
        .def_readwrite("priority", &PatternMetadata::priority)
        .def("is_valid", &PatternMetadata::IsValid);
    
    // ManagedPattern
    py::class_<ManagedPattern>(m, "ManagedPattern")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::string&, PatternCategory>(),
             py::arg("name"), py::arg("pattern_string"), 
             py::arg("category") = PatternCategory::kCustom)
        .def_readwrite("metadata", &ManagedPattern::metadata)
        .def_readwrite("definition", &ManagedPattern::definition)
        .def("is_valid", &ManagedPattern::IsValid)
        .def("get_unique_id", &ManagedPattern::GetUniqueId);
    
    // PatternQuery
    py::class_<PatternQuery>(m, "PatternQuery")
        .def(py::init<>())
        .def_readwrite("category", &PatternQuery::category)
        .def_readwrite("tags", &PatternQuery::tags)
        .def_readwrite("name_contains", &PatternQuery::name_contains)
        .def_readwrite("enabled_only", &PatternQuery::enabled_only)
        .def_readwrite("min_priority", &PatternQuery::min_priority)
        .def_readwrite("max_priority", &PatternQuery::max_priority);
    
    // PatternStatistics
    py::class_<PatternStatistics>(m, "PatternStatistics")
        .def_readonly("total_patterns", &PatternStatistics::total_patterns)
        .def_readonly("enabled_patterns", &PatternStatistics::enabled_patterns)
        .def_readonly("valid_patterns", &PatternStatistics::valid_patterns)
        .def_readonly("invalid_patterns", &PatternStatistics::invalid_patterns)
        .def_readonly("category_counts", &PatternStatistics::category_counts)
        .def_readonly("tag_counts", &PatternStatistics::tag_counts);
    
    // PatternCollection
    py::class_<PatternCollection>(m, "PatternCollection")
        .def(py::init<>())
        .def_readwrite("name", &PatternCollection::name)
        .def_readwrite("description", &PatternCollection::description)
        .def_readwrite("version", &PatternCollection::version)
        .def_readwrite("patterns", &PatternCollection::patterns)
        .def_readwrite("metadata", &PatternCollection::metadata)
        .def("to_json", &PatternCollection::ToJson, py::arg("pretty") = true)
        .def("save_to_file", &PatternCollection::SaveToFile, 
             py::arg("filepath"), py::arg("pretty") = true)
        .def_static("from_json", &PatternCollection::FromJson, py::arg("json_str"))
        .def_static("from_file", &PatternCollection::FromFile, py::arg("filepath"));
    
    // PatternManager
    py::class_<PatternManager>(m, "PatternManager")
        .def(py::init<>())
        // Registration
        .def("register_pattern", py::overload_cast<const ManagedPattern&, bool>(&PatternManager::RegisterPattern),
             py::arg("pattern"), py::arg("overwrite") = false,
             "Register a pattern")
        .def("register_pattern", py::overload_cast<const std::string&, const std::string&, PatternCategory, const std::string&>(&PatternManager::RegisterPattern),
             py::arg("name"), py::arg("pattern_string"), 
             py::arg("category") = PatternCategory::kCustom,
             py::arg("description") = "",
             "Register a simple pattern")
        .def("unregister_pattern", &PatternManager::UnregisterPattern, py::arg("name"))
        .def("clear_patterns", &PatternManager::ClearPatterns)
        .def("clear_patterns_by_category", &PatternManager::ClearPatternsByCategory, py::arg("category"))
        // Retrieval
        .def("get_pattern", &PatternManager::GetPattern, py::arg("name"), py::return_value_policy::reference_internal)
        .def("get_pattern_names", &PatternManager::GetPatternNames)
        // Note: get_all_patterns, get_patterns_by_category, get_patterns_by_tag removed
        // due to binding issues with vector<const ManagedPattern*>
        // Use get_pattern_names() + get_pattern(name) instead
        .def("has_pattern", &PatternManager::HasPattern, py::arg("name"))
        .def("is_pattern_enabled", &PatternManager::IsPatternEnabled, py::arg("name"))
        // State management
        .def("set_pattern_enabled", &PatternManager::SetPatternEnabled, py::arg("name"), py::arg("enabled"))
        .def("set_pattern_priority", &PatternManager::SetPatternPriority, py::arg("name"), py::arg("priority"))
        .def("set_category_enabled", &PatternManager::SetCategoryEnabled, py::arg("category"), py::arg("enabled"))
        // Statistics
        .def("get_statistics", &PatternManager::GetStatistics)
        .def("get_pattern_count", &PatternManager::GetPatternCount)
        .def("get_enabled_pattern_count", &PatternManager::GetEnabledPatternCount)
        .def("print_summary", &PatternManager::PrintSummary)
        .def("get_all_tags", &PatternManager::GetAllTags)
        // Import/Export
        .def("import_patterns", &PatternManager::ImportPatterns, 
             py::arg("collection"), py::arg("overwrite") = false)
        .def("import_patterns_from_json", &PatternManager::ImportPatternsFromJson,
             py::arg("json_str"), py::arg("overwrite") = false)
        .def("import_patterns_from_file", &PatternManager::ImportPatternsFromFile,
             py::arg("filepath"), py::arg("overwrite") = false)
        .def("export_patterns", py::overload_cast<const std::string&>(&PatternManager::ExportPatterns, py::const_),
             py::arg("collection_name") = "patterns")
        .def("export_to_json", &PatternManager::ExportToJson, py::arg("pretty") = true)
        .def("export_to_file", &PatternManager::ExportToFile, 
             py::arg("filepath"), py::arg("pretty") = true)
        // Compiler integration
        .def("get_enabled_pattern_definitions", &PatternManager::GetEnabledPatternDefinitions)
        .def("apply_to_compiler", &PatternManager::ApplyToCompiler, py::arg("compiler"))
        .def("create_compiler", &PatternManager::CreateCompiler);
    
    // PatternRegistry - Simplified to expose only get_manager() for direct access
    py::class_<PatternRegistry>(m, "PatternRegistry")
        .def_static("get_instance", &PatternRegistry::GetInstance, 
                    py::return_value_policy::reference)
        .def("get_manager", py::overload_cast<>(&PatternRegistry::GetManager), 
             py::return_value_policy::reference)
        .def("load_builtin_patterns", &PatternRegistry::LoadBuiltinPatterns)
        .def("load_default_patterns", &PatternRegistry::LoadDefaultPatterns);
    
    // PatternCollections namespace functions
    m.def("get_fusion_patterns", &PatternCollections::GetFusionPatterns, 
          "Get fusion pattern collection");
    m.def("get_optimization_patterns", &PatternCollections::GetOptimizationPatterns,
          "Get optimization pattern collection");
    m.def("get_quantization_patterns", &PatternCollections::GetQuantizationPatterns,
          "Get quantization pattern collection");
    m.def("get_all_builtin_pattern_collections", &PatternCollections::GetAllBuiltinPatterns,
          "Get all built-in pattern collections");
    
    // Global registry access
    m.def("get_pattern_registry", &GetPatternRegistry, 
          py::return_value_policy::reference,
          "Get global pattern registry");
    
    // ========================================================================
    // Utils
    // ========================================================================
    
    // ModelInfo
    py::class_<ModelInfo>(m, "ModelInfo")
        .def_readonly("producer_name", &ModelInfo::producer_name)
        .def_readonly("producer_version", &ModelInfo::producer_version)
        .def_readonly("ir_version", &ModelInfo::ir_version)
        .def_readonly("opset_version", &ModelInfo::opset_version)
        .def_readonly("num_nodes", &ModelInfo::num_nodes)
        .def_readonly("num_inputs", &ModelInfo::num_inputs)
        .def_readonly("num_outputs", &ModelInfo::num_outputs)
        .def_readonly("num_initializers", &ModelInfo::num_initializers)
        .def_readonly("ops_used", &ModelInfo::ops_used);
    
    m.def("load_model", &LoadModel, "Load ONNX model from file");
    m.def("save_model", &SaveModel, "Save model to ONNX file");
    m.def("is_valid_onnx_file", &IsValidONNXFile, "Check if file is valid ONNX");
    m.def("get_onnx_version", &GetONNXVersion, "Get ONNX version");
    m.def("get_model_info", &GetModelInfo, "Get model information");
    m.def("print_model_summary", &PrintModelSummary, "Print model summary");
    
    // ========================================================================
    // Convenience functions
    // ========================================================================
    
    m.def("simplify", [](const std::string& input_path, const std::string& output_path,
                         const SimplifyOptions& options) {
        auto model = LoadModel(input_path);
        if (!model) {
            throw std::runtime_error("Failed to load model: " + input_path);
        }
        
        auto result = Simplifier::Simplify(model, options);
        if (!result.success) {
            throw std::runtime_error("Simplification failed: " + result.error_msg);
        }
        
        if (!SaveModel(model, output_path)) {
            throw std::runtime_error("Failed to save model: " + output_path);
        }
        
        return result;
    }, py::arg("input_path"), py::arg("output_path"), 
       py::arg("options") = SimplifyOptions(),
       "Simplify ONNX model file");
}
