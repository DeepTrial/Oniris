/**
 * @file shape_inference.cpp
 * @brief Shape inference engine implementation with 120+ ONNX operators
 */

#include "passes/shape_inference.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace oniris {
namespace passes {

// =============================================================================
// Shape Helper Functions
// =============================================================================

namespace {

/// Calculate output size for conv/pool operations
inline int64_t CalcConvOutputSize(int64_t input_size, int64_t filter_size,
                                   int64_t stride, int64_t pad_before, 
                                   int64_t pad_after, int64_t dilation) {
    if (input_size < 0) return -1;
    int64_t effective_filter = (filter_size - 1) * dilation + 1;
    return (input_size + pad_before + pad_after - effective_filter) / stride + 1;
}

/// Calculate output size for pooling with ceil mode
inline int64_t CalcPoolOutputSize(int64_t input_size, int64_t kernel,
                                   int64_t stride, int64_t pad_before,
                                   int64_t pad_after, int64_t dilation = 1,
                                   bool ceil_mode = false) {
    if (input_size < 0) return -1;
    int64_t effective_kernel = (kernel - 1) * dilation + 1;
    int64_t numerator = input_size + pad_before + pad_after - effective_kernel;
    if (ceil_mode) {
        return (numerator + stride - 1) / stride + 1;
    }
    return numerator / stride + 1;
}

/// Broadcast two shapes according to NumPy broadcasting rules
Shape BroadcastShapes(const Shape& s1, const Shape& s2) {
    size_t rank1 = s1.NumDims();
    size_t rank2 = s2.NumDims();
    size_t out_rank = std::max(rank1, rank2);
    
    std::vector<Dimension> out_dims(out_rank);
    
    for (size_t i = 0; i < out_rank; ++i) {
        size_t idx1 = rank1 - 1 - i;
        size_t idx2 = rank2 - 1 - i;
        size_t out_idx = out_rank - 1 - i;
        
        const Dimension* d1 = (i < rank1) ? &s1.GetDim(idx1) : nullptr;
        const Dimension* d2 = (i < rank2) ? &s2.GetDim(idx2) : nullptr;
        
        if (!d1 && !d2) {
            out_dims[out_idx] = Dimension(1);
        } else if (!d1) {
            out_dims[out_idx] = *d2;
        } else if (!d2) {
            out_dims[out_idx] = *d1;
        } else if (d1->IsDynamic() || d2->IsDynamic()) {
            if (!d1->IsDynamic() && d1->GetStaticValue() == 1) {
                out_dims[out_idx] = *d2;
            } else if (!d2->IsDynamic() && d2->GetStaticValue() == 1) {
                out_dims[out_idx] = *d1;
            } else {
                out_dims[out_idx] = Dimension();
            }
        } else {
            int64_t v1 = d1->GetStaticValue();
            int64_t v2 = d2->GetStaticValue();
            if (v1 == 1) {
                out_dims[out_idx] = *d2;
            } else if (v2 == 1) {
                out_dims[out_idx] = *d1;
            } else if (v1 == v2) {
                out_dims[out_idx] = *d1;
            } else {
                out_dims[out_idx] = Dimension();
            }
        }
    }
    
    return Shape(out_dims);
}

/// Get axis with negative handling
int64_t GetAxis(int64_t axis, int64_t rank) {
    if (axis < 0) return axis + rank;
    return axis;
}

/// Parse kernel, stride, padding from attributes
void ParseConvAttributes(const InferenceContext& ctx,
                         std::vector<int64_t>& kernel_shape,
                         std::vector<int64_t>& strides,
                         std::vector<int64_t>& pads,
                         std::vector<int64_t>& dilations,
                         int64_t default_dim) {
    auto ks = ctx.GetAttribute<std::vector<int64_t>>("kernel_shape");
    if (ks.has_value()) kernel_shape = *ks;
    
    auto str = ctx.GetAttribute<std::vector<int64_t>>("strides");
    strides = str.value_or(std::vector<int64_t>(default_dim, 1));
    
    auto p = ctx.GetAttribute<std::vector<int64_t>>("pads");
    pads = p.value_or(std::vector<int64_t>(default_dim * 2, 0));
    
    auto d = ctx.GetAttribute<std::vector<int64_t>>("dilations");
    dilations = d.value_or(std::vector<int64_t>(default_dim, 1));
}

}  // anonymous namespace

// =============================================================================
// Math Operators (40+)
// =============================================================================

namespace {

// Element-wise unary - all return same shape as input
InferenceResult InferIdentityUnary(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Unary op requires input");
    }
    return InferenceResult::Success({ctx.input_shapes[0]});
}

// Element-wise binary with broadcasting
InferenceResult InferBroadcastBinary(const InferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return InferenceResult::Error("Binary op requires 2 inputs");
    }
    Shape result = BroadcastShapes(ctx.input_shapes[0], ctx.input_shapes[1]);
    return InferenceResult::Success({result});
}

// Element-wise multi-input
InferenceResult InferElementWise(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Element-wise op requires at least one input");
    }
    Shape result = ctx.input_shapes[0];
    for (size_t i = 1; i < ctx.input_shapes.size(); ++i) {
        result = BroadcastShapes(result, ctx.input_shapes[i]);
    }
    return InferenceResult::Success({result});
}

// MatMul
InferenceResult InferMatMul(const InferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return InferenceResult::Error("MatMul requires 2 inputs");
    }
    const Shape& a = ctx.input_shapes[0];
    const Shape& b = ctx.input_shapes[1];
    
    if (a.NumDims() < 2 || b.NumDims() < 2) {
        return InferenceResult::Error("MatMul inputs need >= 2 dims");
    }
    
    size_t out_rank = std::max(a.NumDims(), b.NumDims());
    std::vector<Dimension> out_dims(out_rank);
    
    size_t batch_dims = out_rank - 2;
    Shape batch_a, batch_b;
    for (size_t i = 0; i < batch_dims; ++i) {
        size_t idx_a = a.NumDims() - 2 - batch_dims + i;
        size_t idx_b = b.NumDims() - 2 - batch_dims + i;
        if (idx_a < a.NumDims()) batch_a.AddDim(a.GetDim(idx_a));
        if (idx_b < b.NumDims()) batch_b.AddDim(b.GetDim(idx_b));
    }
    
    Shape batch_out = BroadcastShapes(batch_a, batch_b);
    for (size_t i = 0; i < batch_dims; ++i) {
        out_dims[i] = batch_out.GetDim(i);
    }
    
    out_dims[out_rank - 2] = a.GetDim(a.NumDims() - 2);
    out_dims[out_rank - 1] = b.GetDim(b.NumDims() - 1);
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Gemm
InferenceResult InferGemm(const InferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return InferenceResult::Error("Gemm requires 2+ inputs");
    }
    const Shape& a = ctx.input_shapes[0];
    const Shape& b = ctx.input_shapes[1];
    
    if (a.NumDims() != 2 || b.NumDims() != 2) {
        return InferenceResult::Error("Gemm inputs must be 2D");
    }
    
    auto transA = ctx.GetAttribute<int64_t>("transA").value_or(0);
    auto transB = ctx.GetAttribute<int64_t>("transB").value_or(0);
    
    Dimension m = transA ? a.GetDim(1) : a.GetDim(0);
    Dimension n = transB ? b.GetDim(0) : b.GetDim(1);
    
    return InferenceResult::Success({Shape({m, n})});
}

// Conv
InferenceResult InferConv(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Conv requires input");
    }
    const Shape& x = ctx.input_shapes[0];
    if (x.NumDims() < 3) {
        return InferenceResult::Error("Conv input needs >= 3 dims");
    }
    
    int64_t spatial_dims = static_cast<int64_t>(x.NumDims()) - 2;
    std::vector<int64_t> kernel_shape, strides, pads, dilations;
    ParseConvAttributes(ctx, kernel_shape, strides, pads, dilations, spatial_dims);
    
    std::vector<Dimension> out_dims(x.NumDims());
    out_dims[0] = x.GetDim(0);
    
    if (ctx.input_shapes.size() >= 2) {
        out_dims[1] = ctx.input_shapes[1].GetDim(0);
    } else {
        out_dims[1] = Dimension();
    }
    
    for (int64_t i = 0; i < spatial_dims; ++i) {
        int64_t input_size = x.GetDim(i + 2).GetStaticValue();
        int64_t k = (i < (int64_t)kernel_shape.size()) ? kernel_shape[i] : 1;
        int64_t s = (i < (int64_t)strides.size()) ? strides[i] : 1;
        int64_t p_before = (2*i < (int64_t)pads.size()) ? pads[2*i] : 0;
        int64_t p_after = (2*i+1 < (int64_t)pads.size()) ? pads[2*i+1] : 0;
        int64_t d = (i < (int64_t)dilations.size()) ? dilations[i] : 1;
        
        if (input_size < 0) {
            out_dims[i + 2] = Dimension();
        } else {
            int64_t out_size = CalcConvOutputSize(input_size, k, s, p_before, p_after, d);
            out_dims[i + 2] = Dimension(out_size);
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Pooling
InferenceResult InferPool(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Pool requires input");
    }
    const Shape& x = ctx.input_shapes[0];
    if (x.NumDims() < 3) {
        return InferenceResult::Error("Pool input needs >= 3 dims");
    }
    
    int64_t spatial_dims = static_cast<int64_t>(x.NumDims()) - 2;
    std::vector<int64_t> kernel_shape, strides, pads, dilations;
    ParseConvAttributes(ctx, kernel_shape, strides, pads, dilations, spatial_dims);
    
    std::vector<Dimension> out_dims(x.GetDims());
    
    for (int64_t i = 0; i < spatial_dims; ++i) {
        int64_t input_size = x.GetDim(i + 2).GetStaticValue();
        int64_t k = (i < (int64_t)kernel_shape.size()) ? kernel_shape[i] : 1;
        int64_t s = (i < (int64_t)strides.size()) ? strides[i] : 1;
        int64_t p_before = (2*i < (int64_t)pads.size()) ? pads[2*i] : 0;
        int64_t p_after = (2*i+1 < (int64_t)pads.size()) ? pads[2*i+1] : 0;
        
        if (input_size < 0) {
            out_dims[i + 2] = Dimension();
        } else {
            int64_t out_size = CalcPoolOutputSize(input_size, k, s, p_before, p_after);
            out_dims[i + 2] = Dimension(out_size);
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Global Pooling
InferenceResult InferGlobalPool(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("GlobalPool requires input");
    }
    const Shape& x = ctx.input_shapes[0];
    std::vector<Dimension> out_dims = x.GetDims();
    for (size_t i = 2; i < out_dims.size(); ++i) {
        out_dims[i] = Dimension(1);
    }
    return InferenceResult::Success({Shape(out_dims)});
}

// Reshape
InferenceResult InferReshape(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Reshape requires input");
    }
    const Shape& data = ctx.input_shapes[0];
    
    auto shape_attr = ctx.GetAttribute<std::vector<int64_t>>("shape");
    if (shape_attr.has_value()) {
        std::vector<Dimension> out_dims;
        int64_t unknown_idx = -1;
        int64_t known_size = 1;
        
        for (size_t i = 0; i < shape_attr->size(); ++i) {
            int64_t dim = (*shape_attr)[i];
            if (dim == 0) {
                if (i < data.NumDims()) {
                    out_dims.push_back(data.GetDim(i));
                    if (!data.GetDim(i).IsDynamic()) {
                        known_size *= data.GetDim(i).GetStaticValue();
                    }
                } else {
                    out_dims.emplace_back(1);
                }
            } else if (dim == -1) {
                out_dims.emplace_back(Dimension());
                unknown_idx = static_cast<int64_t>(i);
            } else {
                out_dims.emplace_back(dim);
                known_size *= dim;
            }
        }
        
        if (unknown_idx >= 0) {
            auto total = data.GetTotalSize();
            if (total.has_value() && known_size > 0) {
                out_dims[unknown_idx] = Dimension(*total / known_size);
            }
        }
        
        return InferenceResult::Success({Shape(out_dims)});
    }
    
    return InferenceResult::Success({Shape({Dimension()})});
}

// Transpose
InferenceResult InferTranspose(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Transpose requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto perm = ctx.GetAttribute<std::vector<int64_t>>("perm");
    if (!perm.has_value()) {
        std::vector<Dimension> out_dims(input.NumDims());
        for (size_t i = 0; i < input.NumDims(); ++i) {
            out_dims[i] = input.GetDim(input.NumDims() - 1 - i);
        }
        return InferenceResult::Success({Shape(out_dims)});
    }
    
    std::vector<Dimension> out_dims(perm->size());
    for (size_t i = 0; i < perm->size(); ++i) {
        int64_t src_idx = (*perm)[i];
        if (src_idx >= 0 && static_cast<size_t>(src_idx) < input.NumDims()) {
            out_dims[i] = input.GetDim(src_idx);
        } else {
            out_dims[i] = Dimension();
        }
    }
    return InferenceResult::Success({Shape(out_dims)});
}

// Squeeze
InferenceResult InferSqueeze(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Squeeze requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto axes_attr = ctx.GetAttribute<std::vector<int64_t>>("axes");
    std::vector<int64_t> axes;
    if (axes_attr.has_value()) {
        axes = *axes_attr;
        for (auto& a : axes) {
            if (a < 0) a += static_cast<int64_t>(input.NumDims());
        }
    }
    
    std::vector<Dimension> out_dims;
    for (size_t i = 0; i < input.NumDims(); ++i) {
        bool squeeze = axes_attr.has_value() ? 
            (std::find(axes.begin(), axes.end(), i) != axes.end()) :
            (input.GetDim(i).GetStaticValue() == 1);
        
        if (!squeeze) {
            out_dims.push_back(input.GetDim(i));
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Unsqueeze
InferenceResult InferUnsqueeze(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Unsqueeze requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto axes_attr = ctx.GetAttribute<std::vector<int64_t>>("axes");
    if (!axes_attr.has_value()) {
        return InferenceResult::Success({Shape({Dimension()})});
    }
    
    std::vector<int64_t> axes = *axes_attr;
    int64_t out_rank = static_cast<int64_t>(input.NumDims()) + static_cast<int64_t>(axes.size());
    
    for (auto& a : axes) {
        if (a < 0) a += out_rank;
    }
    std::sort(axes.begin(), axes.end());
    
    std::vector<Dimension> out_dims;
    size_t input_idx = 0;
    for (int64_t i = 0; i < out_rank; ++i) {
        if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
            out_dims.emplace_back(1);
        } else if (input_idx < input.NumDims()) {
            out_dims.push_back(input.GetDim(input_idx++));
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Flatten
InferenceResult InferFlatten(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Flatten requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto axis_attr = ctx.GetAttribute<int64_t>("axis");
    int64_t axis = axis_attr.value_or(1);
    if (axis < 0) axis += static_cast<int64_t>(input.NumDims());
    
    Dimension d1(1), d2(1);
    bool d1_static = true, d2_static = true;
    int64_t size1 = 1, size2 = 1;
    
    for (int64_t i = 0; i < axis && i < (int64_t)input.NumDims(); ++i) {
        if (input.GetDim(i).IsDynamic()) {
            d1_static = false;
        } else {
            size1 *= input.GetDim(i).GetStaticValue();
        }
    }
    
    for (int64_t i = axis; i < (int64_t)input.NumDims(); ++i) {
        if (input.GetDim(i).IsDynamic()) {
            d2_static = false;
        } else {
            size2 *= input.GetDim(i).GetStaticValue();
        }
    }
    
    if (d1_static) d1 = Dimension(size1);
    if (d2_static) d2 = Dimension(size2);
    
    return InferenceResult::Success({Shape({d1, d2})});
}

// Concat
InferenceResult InferConcat(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Concat requires at least one input");
    }
    
    auto axis_attr = ctx.GetAttribute<int64_t>("axis");
    int64_t axis = axis_attr.value_or(0);
    
    const Shape& first = ctx.input_shapes[0];
    int64_t rank = static_cast<int64_t>(first.NumDims());
    if (axis < 0) axis += rank;
    
    std::vector<Dimension> out_dims(first.GetDims());
    
    bool all_static = !out_dims[axis].IsDynamic();
    int64_t total_size = all_static ? out_dims[axis].GetStaticValue() : 0;
    
    for (size_t i = 1; i < ctx.input_shapes.size(); ++i) {
        const Shape& s = ctx.input_shapes[i];
        if (axis < (int64_t)s.NumDims()) {
            if (s.GetDim(axis).IsDynamic()) {
                all_static = false;
            } else if (all_static) {
                total_size += s.GetDim(axis).GetStaticValue();
            }
        }
    }
    
    if (all_static) {
        out_dims[axis] = Dimension(total_size);
    } else {
        out_dims[axis] = Dimension();
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Split
InferenceResult InferSplit(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Split requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto axis_attr = ctx.GetAttribute<int64_t>("axis");
    int64_t axis = axis_attr.value_or(0);
    if (axis < 0) axis += static_cast<int64_t>(input.NumDims());
    
    auto split_attr = ctx.GetAttribute<std::vector<int64_t>>("split");
    int64_t num_outputs = split_attr.has_value() ? 
                          static_cast<int64_t>(split_attr->size()) : 2;
    
    std::vector<Shape> out_shapes(num_outputs, input);
    
    if (split_attr.has_value()) {
        for (int64_t i = 0; i < num_outputs; ++i) {
            out_shapes[i].GetDim(axis) = Dimension((*split_attr)[i]);
        }
    } else if (!input.GetDim(axis).IsDynamic()) {
        int64_t split_size = input.GetDim(axis).GetStaticValue() / num_outputs;
        for (int64_t i = 0; i < num_outputs; ++i) {
            out_shapes[i].GetDim(axis) = Dimension(split_size);
        }
    }
    
    return InferenceResult::Success(out_shapes);
}

// Slice
InferenceResult InferSlice(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Slice requires input");
    }
    std::vector<Dimension> out_dims(ctx.input_shapes[0].NumDims(), Dimension());
    return InferenceResult::Success({Shape(out_dims)});
}

// Gather
InferenceResult InferGather(const InferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return InferenceResult::Error("Gather requires 2 inputs");
    }
    const Shape& data = ctx.input_shapes[0];
    const Shape& indices = ctx.input_shapes[1];
    
    auto axis_attr = ctx.GetAttribute<int64_t>("axis");
    int64_t axis = axis_attr.value_or(0);
    if (axis < 0) axis += static_cast<int64_t>(data.NumDims());
    
    std::vector<Dimension> out_dims;
    for (int64_t i = 0; i < axis && i < (int64_t)data.NumDims(); ++i) {
        out_dims.push_back(data.GetDim(i));
    }
    for (size_t i = 0; i < indices.NumDims(); ++i) {
        out_dims.push_back(indices.GetDim(i));
    }
    for (int64_t i = axis + 1; i < (int64_t)data.NumDims(); ++i) {
        out_dims.push_back(data.GetDim(i));
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Pad
InferenceResult InferPad(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Pad requires input");
    }
    std::vector<Dimension> out_dims(ctx.input_shapes[0].NumDims(), Dimension());
    return InferenceResult::Success({Shape(out_dims)});
}

// BatchNormalization
InferenceResult InferBatchNormalization(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("BatchNorm requires input");
    }
    std::vector<Shape> outputs = {ctx.input_shapes[0]};
    
    auto training = ctx.GetAttribute<int64_t>("training_mode").value_or(0);
    if (training != 0) {
        if (!ctx.input_shapes[0].GetDims().empty()) {
            Shape stats({ctx.input_shapes[0].GetDim(1)});
            outputs.push_back(stats);
            outputs.push_back(stats);
            outputs.push_back(stats);
            outputs.push_back(stats);
        }
    }
    
    return InferenceResult::Success(outputs);
}

// Reduction
InferenceResult InferReduce(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Reduce requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto axes_attr = ctx.GetAttribute<std::vector<int64_t>>("axes");
    auto keepdims_attr = ctx.GetAttribute<int64_t>("keepdims");
    bool keepdims = keepdims_attr.value_or(1) != 0;
    
    if (!axes_attr.has_value()) {
        if (keepdims) {
            return InferenceResult::Success({Shape(std::vector<Dimension>(input.NumDims(), Dimension(1)))});
        } else {
            return InferenceResult::Success({Shape()});
        }
    }
    
    std::vector<bool> reduce_axis(input.NumDims(), false);
    for (auto a : *axes_attr) {
        if (a < 0) a += static_cast<int64_t>(input.NumDims());
        if (a >= 0 && static_cast<size_t>(a) < input.NumDims()) {
            reduce_axis[a] = true;
        }
    }
    
    std::vector<Dimension> out_dims;
    for (size_t i = 0; i < input.NumDims(); ++i) {
        if (reduce_axis[i]) {
            if (keepdims) out_dims.emplace_back(1);
        } else {
            out_dims.push_back(input.GetDim(i));
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Softmax
InferenceResult InferSoftmax(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Softmax requires input");
    }
    return InferenceResult::Success({ctx.input_shapes[0]});
}

// Cast
InferenceResult InferCast(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Cast requires input");
    }
    return InferenceResult::Success({ctx.input_shapes[0]});
}

// Where
InferenceResult InferWhere(const InferenceContext& ctx) {
    if (ctx.input_shapes.size() < 3) {
        return InferenceResult::Error("Where requires 3 inputs");
    }
    Shape result = BroadcastShapes(ctx.input_shapes[1], ctx.input_shapes[2]);
    result = BroadcastShapes(ctx.input_shapes[0], result);
    return InferenceResult::Success({result});
}

// Shape op
InferenceResult InferShape(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("Shape requires input");
    }
    int64_t rank = static_cast<int64_t>(ctx.input_shapes[0].NumDims());
    return InferenceResult::Success({Shape({Dimension(rank)})});
}

// DepthToSpace
InferenceResult InferDepthToSpace(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("DepthToSpace requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto blocksize = ctx.GetAttribute<int64_t>("blocksize");
    if (!blocksize.has_value()) {
        return InferenceResult::Error("DepthToSpace requires blocksize");
    }
    
    std::vector<Dimension> out_dims = input.GetDims();
    int64_t b = *blocksize;
    
    if (!out_dims[1].IsDynamic()) {
        out_dims[1] = Dimension(out_dims[1].GetStaticValue() / (b * b));
    }
    for (size_t i = 2; i < out_dims.size(); ++i) {
        if (!out_dims[i].IsDynamic()) {
            out_dims[i] = Dimension(out_dims[i].GetStaticValue() * b);
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// SpaceToDepth
InferenceResult InferSpaceToDepth(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("SpaceToDepth requires input");
    }
    const Shape& input = ctx.input_shapes[0];
    
    auto blocksize = ctx.GetAttribute<int64_t>("blocksize");
    if (!blocksize.has_value()) {
        return InferenceResult::Error("SpaceToDepth requires blocksize");
    }
    
    std::vector<Dimension> out_dims = input.GetDims();
    int64_t b = *blocksize;
    
    if (!out_dims[1].IsDynamic()) {
        out_dims[1] = Dimension(out_dims[1].GetStaticValue() * b * b);
    }
    for (size_t i = 2; i < out_dims.size(); ++i) {
        if (!out_dims[i].IsDynamic()) {
            out_dims[i] = Dimension(out_dims[i].GetStaticValue() / b);
        }
    }
    
    return InferenceResult::Success({Shape(out_dims)});
}

// Quantize/Dequantize
InferenceResult InferQuantizeLinear(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("QuantizeLinear requires input");
    }
    return InferenceResult::Success({ctx.input_shapes[0]});
}

InferenceResult InferDequantizeLinear(const InferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return InferenceResult::Error("DequantizeLinear requires input");
    }
    return InferenceResult::Success({ctx.input_shapes[0]});
}

// ConvInteger
InferenceResult InferConvInteger(const InferenceContext& ctx) {
    return InferConv(ctx);
}

// MatMulInteger
InferenceResult InferMatMulInteger(const InferenceContext& ctx) {
    return InferMatMul(ctx);
}

}  // anonymous namespace

// =============================================================================
// Registration Function
// =============================================================================

void ShapeInferenceEngine::InitializeDefaultHandlers() {
    // Math - Unary (all use InferIdentityUnary)
    Register("Abs", InferIdentityUnary);
    Register("Neg", InferIdentityUnary);
    Register("Floor", InferIdentityUnary);
    Register("Ceil", InferIdentityUnary);
    Register("Round", InferIdentityUnary);
    Register("Trunc", InferIdentityUnary);
    Register("Sign", InferIdentityUnary);
    Register("Sqrt", InferIdentityUnary);
    Register("Cbrt", InferIdentityUnary);
    Register("Reciprocal", InferIdentityUnary);
    Register("Exp", InferIdentityUnary);
    Register("Exp2", InferIdentityUnary);
    Register("Expm1", InferIdentityUnary);
    Register("Log", InferIdentityUnary);
    Register("Log10", InferIdentityUnary);
    Register("Log2", InferIdentityUnary);
    Register("Log1p", InferIdentityUnary);
    Register("Sin", InferIdentityUnary);
    Register("Cos", InferIdentityUnary);
    Register("Tan", InferIdentityUnary);
    Register("Asin", InferIdentityUnary);
    Register("Acos", InferIdentityUnary);
    Register("Atan", InferIdentityUnary);
    Register("Sinh", InferIdentityUnary);
    Register("Cosh", InferIdentityUnary);
    Register("Tanh", InferIdentityUnary);
    Register("Asinh", InferIdentityUnary);
    Register("Acosh", InferIdentityUnary);
    Register("Atanh", InferIdentityUnary);
    Register("Erf", InferIdentityUnary);
    Register("Erfc", InferIdentityUnary);
    Register("IsInf", InferIdentityUnary);
    Register("IsNaN", InferIdentityUnary);
    Register("IsFinite", InferIdentityUnary);
    
    // Math - Binary with broadcasting
    Register("Add", InferBroadcastBinary);
    Register("Sub", InferBroadcastBinary);
    Register("Mul", InferBroadcastBinary);
    Register("Div", InferBroadcastBinary);
    Register("Pow", InferBroadcastBinary);
    Register("Mod", InferBroadcastBinary);
    Register("Fmod", InferBroadcastBinary);
    Register("BitShift", InferBroadcastBinary);
    Register("BitwiseAnd", InferBroadcastBinary);
    Register("BitwiseOr", InferBroadcastBinary);
    Register("BitwiseXor", InferBroadcastBinary);
    Register("BitwiseNot", InferIdentityUnary);
    Register("Min", InferBroadcastBinary);
    Register("Max", InferBroadcastBinary);
    Register("Mean", InferBroadcastBinary);
    Register("Sum", InferElementWise);
    
    // Comparison
    Register("Equal", InferBroadcastBinary);
    Register("Greater", InferBroadcastBinary);
    Register("GreaterOrEqual", InferBroadcastBinary);
    Register("Less", InferBroadcastBinary);
    Register("LessOrEqual", InferBroadcastBinary);
    
    // Logical
    Register("And", InferBroadcastBinary);
    Register("Or", InferBroadcastBinary);
    Register("Xor", InferBroadcastBinary);
    Register("Not", InferIdentityUnary);
    
    // Linear Algebra
    Register("MatMul", InferMatMul);
    Register("Gemm", InferGemm);
    
    // Convolution
    Register("Conv", InferConv);
    Register("ConvTranspose", InferConv);  // Simplified
    Register("ConvInteger", InferConvInteger);
    
    // Pooling
    Register("AveragePool", InferPool);
    Register("MaxPool", InferPool);
    Register("GlobalAveragePool", InferGlobalPool);
    Register("GlobalMaxPool", InferGlobalPool);
    
    // Normalization
    Register("BatchNormalization", InferBatchNormalization);
    Register("InstanceNormalization", InferIdentityUnary);
    Register("LayerNormalization", InferIdentityUnary);
    Register("GroupNormalization", InferIdentityUnary);
    
    // Activation
    Register("Relu", InferIdentityUnary);
    Register("LeakyRelu", InferIdentityUnary);
    Register("PRelu", InferIdentityUnary);
    Register("Elu", InferIdentityUnary);
    Register("Celu", InferIdentityUnary);
    Register("ThresholdedRelu", InferIdentityUnary);
    Register("Selu", InferIdentityUnary);
    Register("HardSigmoid", InferIdentityUnary);
    Register("Sigmoid", InferIdentityUnary);
    Register("Shrink", InferIdentityUnary);
    Register("Softplus", InferIdentityUnary);
    Register("Softsign", InferIdentityUnary);
    Register("Clip", InferIdentityUnary);
    Register("HardSwish", InferIdentityUnary);
    Register("Mish", InferIdentityUnary);
    Register("Gelu", InferIdentityUnary);
    Register("QuickGelu", InferIdentityUnary);
    Register("Swish", InferIdentityUnary);
    Register("Softmax", InferSoftmax);
    Register("LogSoftmax", InferSoftmax);
    Register("Hardmax", InferSoftmax);
    
    // Shape Operations
    Register("Reshape", InferReshape);
    Register("Transpose", InferTranspose);
    Register("Squeeze", InferSqueeze);
    Register("Unsqueeze", InferUnsqueeze);
    Register("Flatten", InferFlatten);
    Register("Concat", InferConcat);
    Register("Split", InferSplit);
    Register("Slice", InferSlice);
    Register("Gather", InferGather);
    Register("Pad", InferPad);
    
    // Depth/Space
    Register("DepthToSpace", InferDepthToSpace);
    Register("SpaceToDepth", InferSpaceToDepth);
    
    // Reduction
    Register("ReduceSum", InferReduce);
    Register("ReduceMean", InferReduce);
    Register("ReduceMax", InferReduce);
    Register("ReduceMin", InferReduce);
    Register("ReduceProd", InferReduce);
    Register("ReduceLogSum", InferReduce);
    Register("ReduceLogSumExp", InferReduce);
    Register("ReduceSumSquare", InferReduce);
    Register("ReduceL1", InferReduce);
    Register("ReduceL2", InferReduce);
    Register("ArgMax", InferReduce);
    Register("ArgMin", InferReduce);
    
    // Cast and Utility
    Register("Cast", InferCast);
    Register("CastLike", InferCast);
    Register("Identity", InferIdentityUnary);
    Register("Dropout", InferIdentityUnary);
    Register("Shape", InferShape);
    Register("Where", InferWhere);
    
    // Quantization
    Register("QuantizeLinear", InferQuantizeLinear);
    Register("DequantizeLinear", InferDequantizeLinear);
    Register("MatMulInteger", InferMatMulInteger);
    Register("QLinearConv", InferConv);
    Register("QLinearMatMul", InferMatMul);
    
    ONIRIS_INFO << "Registered 120+ shape inference handlers";
}

// =============================================================================
// Engine Implementation
// =============================================================================

ShapeInferenceEngine::ShapeInferenceEngine() {
    InitializeDefaultHandlers();
}

ShapeInferenceEngine& ShapeInferenceEngine::GetInstance() {
    static ShapeInferenceEngine instance;
    return instance;
}

std::string ShapeInferenceEngine::MakeKey(const std::string& op_type, 
                                           const std::string& domain) {
    if (domain.empty() || domain == "ai.onnx") {
        return op_type;
    }
    return domain + "::" + op_type;
}

void ShapeInferenceEngine::Register(const std::string& op_type, ShapeInferFunc func,
                                     const std::string& domain) {
    std::string key = MakeKey(op_type, domain);
    handlers_[key] = func;
    ONIRIS_DEBUG << "Registered shape inference for " << key;
}

void ShapeInferenceEngine::Unregister(const std::string& op_type, 
                                       const std::string& domain) {
    std::string key = MakeKey(op_type, domain);
    handlers_.erase(key);
}

bool ShapeInferenceEngine::HasHandler(const std::string& op_type,
                                       const std::string& domain) const {
    std::string key = MakeKey(op_type, domain);
    return handlers_.find(key) != handlers_.end();
}

bool ShapeInferenceEngine::GetValueInfo(const Graph& graph, const std::string& name,
                                         Shape* shape, DataType* dtype) {
    for (const auto& input : graph.GetInputs()) {
        if (input.name == name) {
            if (shape) *shape = input.shape;
            if (dtype) *dtype = input.dtype;
            return true;
        }
    }
    
    auto vi = graph.GetValueInfo(name);
    if (vi != nullptr) {
        if (shape) *shape = vi->shape;
        if (dtype) *dtype = vi->dtype;
        return true;
    }
    
    auto ct = graph.GetConstant(name);
    if (ct != nullptr) {
        if (shape) *shape = ct->shape;
        if (dtype) *dtype = ct->dtype;
        return true;
    }
    
    auto init = graph.GetInitializers().find(name);
    if (init != graph.GetInitializers().end()) {
        if (shape) *shape = init->second.GetShape();
        if (dtype) *dtype = init->second.GetDataType();
        return true;
    }
    
    return false;
}

InferenceResult ShapeInferenceEngine::InferNode(const std::shared_ptr<Node>& node,
                                                 const Graph& graph) {
    std::string key = MakeKey(node->GetOpType(), node->GetDomain());
    
    auto it = handlers_.find(key);
    if (it == handlers_.end()) {
        return InferenceResult::Error("No shape inference for " + key);
    }
    
    InferenceContext ctx;
    ctx.graph = &graph;
    ctx.attributes = &node->GetAttributes();
    
    for (const auto& input_name : node->GetInputs()) {
        if (input_name.empty()) {
            ctx.input_shapes.push_back(Shape());
            ctx.input_dtypes.push_back(DataType::kUnknown);
            continue;
        }
        
        Shape shape;
        DataType dtype;
        if (GetValueInfo(graph, input_name, &shape, &dtype)) {
            ctx.input_shapes.push_back(shape);
            ctx.input_dtypes.push_back(dtype);
        } else {
            ctx.input_shapes.push_back(Shape({Dimension()}));
            ctx.input_dtypes.push_back(DataType::kUnknown);
        }
    }
    
    return it->second(ctx);
}

bool ShapeInferenceEngine::InferGraph(const std::shared_ptr<Graph>& graph,
                                       bool fail_on_unknown) {
    if (!graph) {
        ONIRIS_ERROR << "Cannot infer shapes for null graph";
        return false;
    }
    
    bool all_success = true;
    auto sorted_nodes = graph->TopologicalSort();
    
    for (const auto& node : sorted_nodes) {
        auto result = InferNode(node, *graph);
        
        if (!result.success) {
            ONIRIS_WARNING << "Shape inference failed for " << node->GetName()
                          << " (" << node->GetOpType() << "): " << result.error_msg;
            all_success = false;
            
            if (fail_on_unknown) {
                return false;
            }
            continue;
        }
        
        for (size_t i = 0; i < result.output_shapes.size() && i < node->GetOutputs().size(); ++i) {
            const std::string& output_name = node->GetOutputs()[i];
            const Shape& shape = result.output_shapes[i];
            
            auto vi = graph->GetValueInfo(output_name);
            if (vi != nullptr) {
                vi->shape = shape;
            } else {
                ValueInfo new_vi;
                new_vi.name = output_name;
                new_vi.shape = shape;
                if (i < result.output_dtypes.size()) {
                    new_vi.dtype = result.output_dtypes[i];
                }
                graph->SetValueInfo(output_name, new_vi);
            }
            
            node->SetOutputShape(i, shape);
        }
    }
    
    return all_success;
}

bool ShapeInferenceEngine::InferModel(const std::shared_ptr<Model>& model,
                                       bool fail_on_unknown) {
    if (!model) {
        ONIRIS_ERROR << "Cannot infer shapes for null model";
        return false;
    }
    return InferGraph(model->GetGraph(), fail_on_unknown);
}

std::vector<std::string> ShapeInferenceEngine::GetSupportedOps() const {
    std::vector<std::string> ops;
    ops.reserve(handlers_.size());
    for (const auto& [key, _] : handlers_) {
        ops.push_back(key);
    }
    return ops;
}

}  // namespace passes
}  // namespace oniris
