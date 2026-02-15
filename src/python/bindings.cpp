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
        .value("UNKNOWN", DataType::kUnknown)
        .value("FLOAT32", DataType::kFloat32)
        .value("UINT8", DataType::kUint8)
        .value("INT8", DataType::kInt8)
        .value("UINT16", DataType::kUint16)
        .value("INT16", DataType::kInt16)
        .value("INT32", DataType::kInt32)
        .value("INT64", DataType::kInt64)
        .value("STRING", DataType::kString)
        .value("BOOL", DataType::kBool)
        .value("FLOAT16", DataType::kFloat16)
        .value("FLOAT64", DataType::kFloat64)
        .value("UINT32", DataType::kUint32)
        .value("UINT64", DataType::kUint64)
        .value("BFLOAT16", DataType::kBFloat16)
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
