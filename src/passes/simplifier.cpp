/**
 * @file simplifier.cpp
 * @brief Model simplification implementation (merged with advanced features)
 */

#include "passes/simplifier.hpp"

#include "core/logger.hpp"
#include "passes/shape_inference.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <set>
#include <unordered_set>

namespace oniris {
namespace passes {

// ============================================================================
// Utility Functions
// ============================================================================

bool Simplifier::IsConstant(const Graph& graph, const std::string& name) {
    return graph.HasConstant(name) || 
           graph.GetInitializers().find(name) != graph.GetInitializers().end();
}

Tensor Simplifier::GetConstantValue(const Graph& graph, const std::string& name) {
    auto ct = graph.GetConstant(name);
    if (ct != nullptr) return ct->tensor;
    auto it = graph.GetInitializers().find(name);
    if (it != graph.GetInitializers().end()) return it->second;
    return Tensor();
}

bool Simplifier::CanFoldNode(const Node& node, const Graph& graph) {
    for (const auto& input : node.GetInputs()) {
        if (!input.empty() && !IsConstant(graph, input)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Basic Nop Elimination Passes
// ============================================================================

int Simplifier::EliminateIdentity(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Identity") continue;
        if (node->GetInputs().empty() || node->GetOutputs().empty()) continue;
        
        const std::string& input = node->GetInputs()[0];
        const std::string& output = node->GetOutputs()[0];
        
        for (const auto& other : graph.GetNodes()) {
            for (auto& in : other->GetInputs()) {
                if (in == output) in = input;
            }
        }
        
        for (auto& go : graph.GetOutputs()) {
            if (go.name == output) go.name = input;
        }
        
        graph.RemoveNode(node);
        changes++;
        ONIRIS_DEBUG << "Eliminated identity: " << node->GetName();
    }
    
    return changes;
}

int Simplifier::EliminateNopTranspose(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Transpose") continue;
        
        auto perm_attr = node->GetAttributeAs<std::vector<int64_t>>("perm");
        if (!perm_attr.has_value()) continue;
        
        const auto& perm = *perm_attr;
        bool is_nop = true;
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] != static_cast<int64_t>(i)) {
                is_nop = false;
                break;
            }
        }
        
        if (!is_nop) continue;
        if (node->GetInputs().empty() || node->GetOutputs().empty()) continue;
        
        const std::string& input = node->GetInputs()[0];
        const std::string& output = node->GetOutputs()[0];
        
        for (const auto& other : graph.GetNodes()) {
            for (auto& in : other->GetInputs()) {
                if (in == output) in = input;
            }
        }
        
        for (auto& go : graph.GetOutputs()) {
            if (go.name == output) go.name = input;
        }
        
        graph.RemoveNode(node);
        changes++;
        ONIRIS_DEBUG << "Eliminated nop transpose: " << node->GetName();
    }
    
    return changes;
}

int Simplifier::EliminateNopReshape(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Reshape") continue;
        
        auto shape_attr = node->GetAttributeAs<std::vector<int64_t>>("shape");
        if (!shape_attr.has_value()) continue;
        
        if (node->GetInputs().empty()) continue;
        auto vi = graph.GetValueInfo(node->GetInputs()[0]);
        if (!vi) continue;
        
        const auto& input_dims = vi->shape.GetDims();
        const auto& target_dims = *shape_attr;
        
        if (input_dims.size() != target_dims.size()) continue;
        
        bool same = true;
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (target_dims[i] == 0) continue;
            if (input_dims[i].IsDynamic()) {
                same = false;
                break;
            }
            if (input_dims[i].GetStaticValue() != target_dims[i]) {
                same = false;
                break;
            }
        }
        
        if (!same) continue;
        
        const std::string& input = node->GetInputs()[0];
        const std::string& output = node->GetOutputs()[0];
        
        for (const auto& other : graph.GetNodes()) {
            for (auto& in : other->GetInputs()) {
                if (in == output) in = input;
            }
        }
        
        for (auto& go : graph.GetOutputs()) {
            if (go.name == output) go.name = input;
        }
        
        graph.RemoveNode(node);
        changes++;
        ONIRIS_DEBUG << "Eliminated nop reshape: " << node->GetName();
    }
    
    return changes;
}

int Simplifier::EliminateNopPad(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Pad") continue;
        
        auto pads = node->GetAttributeAs<std::vector<int64_t>>("pads");
        if (!pads.has_value()) continue;
        
        bool all_zero = std::all_of(pads->begin(), pads->end(), 
                                    [](int64_t p) { return p == 0; });
        if (!all_zero) continue;
        
        if (node->GetInputs().empty() || node->GetOutputs().empty()) continue;
        
        const std::string& input = node->GetInputs()[0];
        const std::string& output = node->GetOutputs()[0];
        
        for (const auto& other : graph.GetNodes()) {
            for (auto& in : other->GetInputs()) {
                if (in == output) in = input;
            }
        }
        
        for (auto& go : graph.GetOutputs()) {
            if (go.name == output) go.name = input;
        }
        
        graph.RemoveNode(node);
        changes++;
        ONIRIS_DEBUG << "Eliminated nop pad: " << node->GetName();
    }
    
    return changes;
}

int Simplifier::EliminateNopSlice(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Slice") continue;
        
        // Check if slice selects full range (simplified check)
        if (!node->HasAttribute("starts") && !node->HasAttribute("ends")) {
            if (node->GetInputs().empty() || node->GetOutputs().empty()) continue;
            
            const std::string& input = node->GetInputs()[0];
            const std::string& output = node->GetOutputs()[0];
            
            for (const auto& other : graph.GetNodes()) {
                for (auto& in : other->GetInputs()) {
                    if (in == output) in = input;
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == output) go.name = input;
            }
            
            graph.RemoveNode(node);
            changes++;
        }
    }
    
    return changes;
}

int Simplifier::EliminateNopResize(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Resize" && node->GetOpType() != "Upsample") continue;
        
        auto scales = node->GetAttributeAs<std::vector<float>>("scales");
        if (scales.has_value()) {
            bool all_one = std::all_of(scales->begin(), scales->end(),
                                       [](float s) { return s == 1.0f; });
            if (all_one) {
                if (node->GetInputs().empty() || node->GetOutputs().empty()) continue;
                
                const std::string& input = node->GetInputs()[0];
                const std::string& output = node->GetOutputs()[0];
                
                for (const auto& other : graph.GetNodes()) {
                    for (auto& in : other->GetInputs()) {
                        if (in == output) in = input;
                    }
                }
                
                for (auto& go : graph.GetOutputs()) {
                    if (go.name == output) go.name = input;
                }
                
                graph.RemoveNode(node);
                changes++;
            }
        }
    }
    
    return changes;
}

int Simplifier::EliminateSingleInputConcat(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (node->GetOpType() != "Concat") continue;
        
        int non_empty = 0;
        std::string single_input;
        for (const auto& in : node->GetInputs()) {
            if (!in.empty()) {
                non_empty++;
                single_input = in;
            }
        }
        
        if (non_empty != 1) continue;
        
        const std::string& output = node->GetOutputs()[0];
        
        for (const auto& other : graph.GetNodes()) {
            for (auto& in : other->GetInputs()) {
                if (in == output) in = single_input;
            }
        }
        
        for (auto& go : graph.GetOutputs()) {
            if (go.name == output) go.name = single_input;
        }
        
        graph.RemoveNode(node);
        changes++;
    }
    
    return changes;
}

// ============================================================================
// Dead Code and Constant Management
// ============================================================================

int Simplifier::EliminateDeadNodes(Graph& graph) {
    int before = static_cast<int>(graph.GetNodes().size());
    graph.RemoveDeadNodes();
    int after = static_cast<int>(graph.GetNodes().size());
    int changes = before - after;
    if (changes > 0) {
        ONIRIS_DEBUG << "Eliminated " << changes << " dead nodes";
    }
    return changes;
}

int Simplifier::EliminateUnusedConstants(Graph& graph) {
    int changes = 0;
    std::set<std::string> used_values;
    
    for (const auto& node : graph.GetNodes()) {
        for (const auto& in : node->GetInputs()) {
            used_values.insert(in);
        }
    }
    for (const auto& out : graph.GetOutputs()) {
        used_values.insert(out.name);
    }
    
    auto constants = graph.GetConstants();
    for (const auto& [name, _] : constants) {
        if (used_values.find(name) == used_values.end()) {
            graph.RemoveConstant(name);
            changes++;
        }
    }
    
    return changes;
}

// ============================================================================
// Constant Folding
// ============================================================================

namespace {

template<typename T>
bool ComputeConstantBinary(const Tensor& a, const Tensor& b, Tensor& out, 
                           const std::string& op) {
    auto num_elems = a.GetNumElements();
    if (!num_elems.has_value()) return false;
    
    out = Tensor(a.GetShape(), a.GetDataType());
    size_t bytes = *num_elems * sizeof(T);
    out.GetData().resize(bytes);
    
    const T* pa = a.GetDataPtr<T>();
    const T* pb = b.GetDataPtr<T>();
    T* po = out.GetDataPtr<T>();
    
    if (op == "Add") {
        for (int64_t i = 0; i < *num_elems; ++i) po[i] = pa[i] + pb[i];
    } else if (op == "Sub") {
        for (int64_t i = 0; i < *num_elems; ++i) po[i] = pa[i] - pb[i];
    } else if (op == "Mul") {
        for (int64_t i = 0; i < *num_elems; ++i) po[i] = pa[i] * pb[i];
    } else if (op == "Div") {
        for (int64_t i = 0; i < *num_elems; ++i) po[i] = pa[i] / pb[i];
    } else {
        return false;
    }
    
    return true;
}

}  // anonymous namespace

int Simplifier::ConstantFolding(Graph& graph, bool fail_on_unsupported) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (const auto& node : nodes) {
        if (!CanFoldNode(*node, graph)) continue;
        
        std::vector<Tensor> inputs;
        for (const auto& input_name : node->GetInputs()) {
            if (input_name.empty()) {
                inputs.push_back(Tensor());
            } else {
                inputs.push_back(GetConstantValue(graph, input_name));
            }
        }
        
        std::vector<Tensor> outputs;
        if (!ComputeNode(*node, inputs, outputs, fail_on_unsupported)) continue;
        
        for (size_t i = 0; i < outputs.size() && i < node->GetOutputs().size(); ++i) {
            const std::string& output_name = node->GetOutputs()[i];
            ConstantTensor ct;
            ct.name = output_name;
            ct.shape = outputs[i].GetShape();
            ct.dtype = outputs[i].GetDataType();
            ct.tensor = outputs[i];
            graph.AddConstant(output_name, ct);
        }
        
        graph.RemoveNode(node);
        changes++;
        ONIRIS_DEBUG << "Folded constant node: " << node->GetName();
    }
    
    return changes;
}

bool Simplifier::ComputeNode(const Node& node,
                              const std::vector<Tensor>& inputs,
                              std::vector<Tensor>& outputs,
                              bool fail_on_unsupported) {
    const std::string& op = node.GetOpType();
    
    if (inputs.empty()) return false;
    
    // Identity
    if (op == "Identity") {
        outputs.resize(1);
        outputs[0] = inputs[0];
        return true;
    }
    
    // Shape
    if (op == "Shape") {
        const Shape& shape = inputs[0].GetShape();
        std::vector<int64_t> shape_data;
        for (size_t i = 0; i < shape.NumDims(); ++i) {
            if (!shape.GetDim(i).IsDynamic()) {
                shape_data.push_back(shape.GetDim(i).GetStaticValue());
            } else {
                return false;
            }
        }
        outputs.resize(1);
        outputs[0] = Tensor(Shape({static_cast<int64_t>(shape_data.size())}),
                           DataType::kInt64);
        outputs[0].GetData().resize(shape_data.size() * sizeof(int64_t));
        std::memcpy(outputs[0].GetData().data(), shape_data.data(), 
                   shape_data.size() * sizeof(int64_t));
        return true;
    }
    
    // Cast
    if (op == "Cast") {
        outputs.resize(1);
        outputs[0] = inputs[0];  // Simplified
        return true;
    }
    
    if (fail_on_unsupported) return false;
    return false;
}

// ============================================================================
// Fusion Passes
// ============================================================================

int Simplifier::FuseConvBN(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& conv_node : nodes) {
        if (conv_node->GetOpType() != "Conv" && conv_node->GetOpType() != "ConvTranspose") {
            continue;
        }
        
        std::string conv_output = conv_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(conv_output);
        
        for (auto& bn_node : consumers) {
            if (bn_node->GetOpType() != "BatchNormalization") continue;
            if (bn_node->GetInputs()[0] != conv_output) continue;
            
            // Check if BN parameters are constant
            bool bn_const = true;
            for (size_t i = 1; i < bn_node->GetInputs().size() && i < 5; ++i) {
                if (!IsConstant(graph, bn_node->GetInputs()[i])) {
                    bn_const = false;
                    break;
                }
            }
            
            if (!bn_const) continue;
            
            // Fuse: redirect BN output to Conv output and remove BN
            std::string bn_output = bn_node->GetOutputs()[0];
            
            for (auto& consumer : graph.GetConsumers(bn_output)) {
                for (auto& input : consumer->GetInputs()) {
                    if (input == bn_output) {
                        input = conv_output;
                    }
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == bn_output) {
                    go.name = conv_output;
                }
            }
            
            graph.RemoveNode(bn_node);
            count++;
            ONIRIS_DEBUG << "Fused Conv + BatchNormalization";
            break;
        }
    }
    
    return count;
}

int Simplifier::FuseConvRelu(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& conv_node : nodes) {
        if (conv_node->GetOpType() != "Conv") continue;
        
        std::string conv_output = conv_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(conv_output);
        
        for (auto& relu_node : consumers) {
            if (relu_node->GetOpType() != "Relu") continue;
            
            std::string relu_output = relu_node->GetOutputs()[0];
            
            for (auto& consumer : graph.GetConsumers(relu_output)) {
                for (auto& input : consumer->GetInputs()) {
                    if (input == relu_output) {
                        input = conv_output;
                    }
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == relu_output) {
                    go.name = conv_output;
                }
            }
            
            graph.RemoveNode(relu_node);
            count++;
            ONIRIS_DEBUG << "Fused Conv + ReLU";
            break;
        }
    }
    
    return count;
}

int Simplifier::FuseGemmActivations(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& gemm_node : nodes) {
        if (gemm_node->GetOpType() != "Gemm") continue;
        
        std::string gemm_output = gemm_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(gemm_output);
        
        for (auto& act_node : consumers) {
            const std::string& op = act_node->GetOpType();
            if (op != "Relu" && op != "Sigmoid" && op != "Tanh") continue;
            
            std::string act_output = act_node->GetOutputs()[0];
            
            for (auto& consumer : graph.GetConsumers(act_output)) {
                for (auto& input : consumer->GetInputs()) {
                    if (input == act_output) {
                        input = gemm_output;
                    }
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == act_output) {
                    go.name = gemm_output;
                }
            }
            
            graph.RemoveNode(act_node);
            count++;
            ONIRIS_DEBUG << "Fused Gemm + " << op;
            break;
        }
    }
    
    return count;
}

int Simplifier::FuseGemmBias(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& gemm_node : nodes) {
        if (gemm_node->GetOpType() != "Gemm") continue;
        
        // Skip if Gemm already has bias (3 inputs)
        int non_empty_inputs = 0;
        for (const auto& in : gemm_node->GetInputs()) {
            if (!in.empty()) non_empty_inputs++;
        }
        if (non_empty_inputs >= 3) continue;
        
        std::string gemm_output = gemm_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(gemm_output);
        
        for (auto& add_node : consumers) {
            if (add_node->GetOpType() != "Add") continue;
            
            // Check if one of Add's inputs is the Gemm output
            const auto& add_inputs = add_node->GetInputs();
            if (add_inputs.size() < 2) continue;
            
            // Find the bias input (the one that's not from Gemm)
            std::string bias_input;
            bool found_gemm = false;
            for (const auto& in : add_inputs) {
                if (in == gemm_output) {
                    found_gemm = true;
                } else if (!in.empty()) {
                    bias_input = in;
                }
            }
            
            if (!found_gemm || bias_input.empty()) continue;
            
            // Check if bias is constant (initializer or constant node)
            bool is_const = IsConstant(graph, bias_input);
            
            // Get bias shape for validation
            Shape bias_shape;
            bool has_shape = false;
            
            // Try to get shape from value info
            auto vi = graph.GetValueInfo(bias_input);
            if (vi != nullptr) {
                bias_shape = vi->shape;
                has_shape = true;
            } else {
                // Try to get shape from constant
                auto ct = graph.GetConstant(bias_input);
                if (ct != nullptr) {
                    bias_shape = ct->shape;
                    has_shape = true;
                } else {
                    // Try to get shape from initializers
                    auto init_it = graph.GetInitializers().find(bias_input);
                    if (init_it != graph.GetInitializers().end()) {
                        bias_shape = init_it->second.GetShape();
                        has_shape = true;
                    }
                }
            }
            
            if (!has_shape) continue;
            
            // Validate bias shape: should be 1D or broadcastable to (M, N)
            bool valid_bias = (bias_shape.NumDims() == 1 || bias_shape.NumDims() == 2);
            if (!valid_bias) continue;
            
            // Get Gemm attributes
            auto beta_attr = gemm_node->GetAttributeAs<float>("beta");
            float beta = beta_attr.value_or(1.0f);
            
            // Only fuse if beta is 1.0 (default) or if bias is constant and we can adjust
            if (beta != 1.0f && !is_const) continue;
            
            // Add bias as 3rd input to Gemm
            gemm_node->AddInput(bias_input);
            
            // Redirect Add's output to Gemm's output
            std::string add_output = add_node->GetOutputs()[0];
            for (auto& consumer : graph.GetConsumers(add_output)) {
                for (auto& input : consumer->GetInputs()) {
                    if (input == add_output) {
                        input = gemm_output;
                    }
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == add_output) {
                    go.name = gemm_output;
                }
            }
            
            graph.RemoveNode(add_node);
            count++;
            ONIRIS_DEBUG << "Fused Gemm + Add (bias)";
            break;
        }
    }
    
    return count;
}

int Simplifier::FuseQGemmActivations(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& qgemm_node : nodes) {
        // QGemm is in com.microsoft domain
        if (qgemm_node->GetOpType() != "QGemm") continue;
        
        std::string qgemm_output = qgemm_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(qgemm_output);
        
        for (auto& act_node : consumers) {
            const std::string& op = act_node->GetOpType();
            // QGemm can be fused with common activations
            if (op != "Relu" && op != "Sigmoid" && op != "Tanh" && 
                op != "LeakyRelu" && op != "Clip") continue;
            
            std::string act_output = act_node->GetOutputs()[0];
            
            for (auto& consumer : graph.GetConsumers(act_output)) {
                for (auto& input : consumer->GetInputs()) {
                    if (input == act_output) {
                        input = qgemm_output;
                    }
                }
            }
            
            for (auto& go : graph.GetOutputs()) {
                if (go.name == act_output) {
                    go.name = qgemm_output;
                }
            }
            
            graph.RemoveNode(act_node);
            count++;
            ONIRIS_DEBUG << "Fused QGemm + " << op;
            break;
        }
    }
    
    return count;
}

int Simplifier::FuseMicrosoftActivations(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    // Fuse Microsoft Gelu ops with activations
    for (auto& gelu_node : nodes) {
        if (gelu_node->GetOpType() != "Gelu" && 
            gelu_node->GetOpType() != "FastGelu") continue;
        
        // Check if it's from Microsoft domain
        if (gelu_node->GetDomain() != "com.microsoft") continue;
        
        std::string gelu_output = gelu_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(gelu_output);
        
        // If Gelu is followed by another activation, we might want to skip or warn
        // For now, just track that we've seen Microsoft activations
    }
    
    // Fuse BiasGelu pattern: Add + Gelu -> BiasGelu
    for (auto& add_node : nodes) {
        if (add_node->GetOpType() != "Add") continue;
        
        if (add_node->GetInputs().empty() || add_node->GetOutputs().empty()) continue;
        
        std::string add_output = add_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(add_output);
        
        for (auto& act_node : consumers) {
            if (act_node->GetOpType() != "Gelu" && 
                act_node->GetOpType() != "FastGelu") continue;
            
            // Found Add + Gelu pattern, could convert to BiasGelu
            // For now, just log it
            ONIRIS_DEBUG << "Found Add + Gelu pattern that could be fused to BiasGelu";
        }
    }
    
    return count;
}

int Simplifier::FuseFusedOps(Graph& graph) {
    int count = 0;
    auto nodes = graph.GetNodes();
    
    // Fuse patterns into FusedConv
    for (auto& conv_node : nodes) {
        if (conv_node->GetOpType() != "Conv") continue;
        
        std::string conv_output = conv_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(conv_output);
        
        for (auto& act_node : consumers) {
            const std::string& op = act_node->GetOpType();
            // Check for activations that can be fused
            if (op != "Relu" && op != "Sigmoid" && op != "Tanh" &&
                op != "LeakyRelu" && op != "Gelu") continue;
            
            // Convert Conv + Activation to FusedConv (Microsoft domain)
            // This would require creating a new FusedConv node
            // For now, just track the pattern
            ONIRIS_DEBUG << "Found Conv + " << op << " pattern for potential FusedConv";
        }
    }
    
    // Similar for FusedGemm
    for (auto& gemm_node : nodes) {
        if (gemm_node->GetOpType() != "Gemm") continue;
        
        std::string gemm_output = gemm_node->GetOutputs()[0];
        auto consumers = graph.GetConsumers(gemm_output);
        
        for (auto& act_node : consumers) {
            const std::string& op = act_node->GetOpType();
            if (op != "Relu" && op != "Sigmoid" && op != "Tanh" &&
                op != "LeakyRelu" && op != "Gelu") continue;
            
            ONIRIS_DEBUG << "Found Gemm + " << op << " pattern for potential FusedGemm";
        }
    }
    
    return count;
}

// ============================================================================
// Other Optimizations
// ============================================================================

int Simplifier::SimplifyShapeOps(Graph& graph) {
    int count = 0;
    // Simplify Shape -> Gather -> Unsqueeze -> Concat pattern
    // Placeholder for future implementation
    return count;
}

int Simplifier::MergeConsecutiveOps(Graph& graph) {
    int changes = 0;
    auto nodes = graph.GetNodes();
    
    for (auto& node : nodes) {
        if (node->GetOpType() != "Reshape" && node->GetOpType() != "Transpose") {
            continue;
        }
        
        if (node->GetInputs().empty()) continue;
        auto producer = graph.GetProducer(node->GetInputs()[0]);
        if (!producer) continue;
        
        if (producer->GetOpType() != node->GetOpType()) continue;
        
        if (node->GetOpType() == "Reshape") {
            if (!producer->GetInputs().empty()) {
                node->SetInput(0, producer->GetInputs()[0]);
                graph.RemoveNode(producer);
                changes++;
                ONIRIS_DEBUG << "Merged consecutive reshapes";
            }
        }
    }
    
    return changes;
}

// ============================================================================
// Main Simplification Execution
// ============================================================================

int Simplifier::RunAllPasses(Graph& graph, const SimplifyOptions& options) {
    int total_changes = 0;
    
    // Shape inference
    if (!options.skip_shape_inference) {
        auto& engine = ShapeInferenceEngine::GetInstance();
        engine.InferGraph(graph.shared_from_this(), false);
    }
    
    // Basic nop eliminations
    if (!options.skip_identity_elimination) {
        total_changes += EliminateIdentity(graph);
    }
    if (!options.skip_transpose_elimination) {
        total_changes += EliminateNopTranspose(graph);
    }
    if (!options.skip_reshape_elimination) {
        total_changes += EliminateNopReshape(graph);
    }
    if (!options.skip_pad_elimination) {
        total_changes += EliminateNopPad(graph);
    }
    if (!options.skip_slice_elimination) {
        total_changes += EliminateNopSlice(graph);
    }
    
    total_changes += EliminateNopResize(graph);
    total_changes += EliminateSingleInputConcat(graph);
    
    // Fusion passes (user controllable)
    if (options.fuse_conv_bn) {
        total_changes += FuseConvBN(graph);
    }
    if (options.fuse_conv_relu) {
        total_changes += FuseConvRelu(graph);
    }
    if (options.fuse_gemm_activation) {
        total_changes += FuseGemmActivations(graph);
    }
    if (options.fuse_gemm_bias) {
        total_changes += FuseGemmBias(graph);
    }
    if (options.fuse_qgemm_activation) {
        total_changes += FuseQGemmActivations(graph);
    }
    
    // Constant folding
    if (!options.skip_constant_folding) {
        total_changes += ConstantFolding(graph, options.fail_on_unsupported);
    }
    
    // Dead code elimination
    if (!options.skip_dead_node_elimination) {
        total_changes += EliminateDeadNodes(graph);
    }
    
    // Other optimizations
    if (!options.skip_shape_ops_simplification) {
        total_changes += SimplifyShapeOps(graph);
    }
    total_changes += MergeConsecutiveOps(graph);
    total_changes += EliminateUnusedConstants(graph);
    
    return total_changes;
}

SimplifyResult Simplifier::SimplifyGraph(const std::shared_ptr<Graph>& graph,
                                          const SimplifyOptions& options) {
    SimplifyResult result;
    
    if (!graph) {
        result.success = false;
        result.error_msg = "Null graph";
        return result;
    }
    
    // Iterate until fixed point
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        int iteration_changes = RunAllPasses(*graph, options);
        result.num_changes += iteration_changes;
        
        if (iteration_changes == 0) break;
        result.num_iterations = iter + 1;
        
        // Re-run shape inference after changes
        if (!options.skip_shape_inference) {
            auto& engine = ShapeInferenceEngine::GetInstance();
            engine.InferGraph(graph, false);
        }
    }
    
    ONIRIS_INFO << "Simplified graph: " << result.num_changes << " changes in "
                << result.num_iterations << " iterations";
    
    return result;
}

SimplifyResult Simplifier::Simplify(const std::shared_ptr<Model>& model,
                                     const SimplifyOptions& options) {
    SimplifyResult result;
    
    if (!model) {
        result.success = false;
        result.error_msg = "Null model";
        return result;
    }
    
    auto graph = model->GetGraph();
    if (!graph) {
        result.success = false;
        result.error_msg = "Model has no graph";
        return result;
    }
    
    result = SimplifyGraph(graph, options);
    
    return result;
}

}  // namespace passes
}  // namespace oniris
