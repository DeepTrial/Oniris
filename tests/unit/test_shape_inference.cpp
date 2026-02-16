/**
 * @file test_shape_inference.cpp
 * @brief Unit tests for shape inference (120+ operators)
 */

#include "passes/shape_inference.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace oniris;
using namespace oniris::passes;

// ============================================================================
// Math Operators Tests
// ============================================================================

TEST(ShapeInference, ElementWiseUnary) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "x";
    input.shape = Shape({2, 3, 4});
    input.dtype = DataType::kFloat32;
    graph.AddInput(input);
    
    std::vector<std::string> ops = {
        "Abs", "Neg", "Floor", "Ceil", "Round", "Trunc", "Sign",
        "Sqrt", "Cbrt", "Reciprocal", "Exp", "Exp2", "Expm1",
        "Log", "Log10", "Log2", "Log1p",
        "Sin", "Cos", "Tan", "Asin", "Acos", "Atan",
        "Sinh", "Cosh", "Tanh", "Asinh", "Acosh", "Atanh",
        "Erf", "Erfc"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->AddInput("x");
        node->AddOutput("y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        if (result.success) {
            EXPECT_EQ(result.output_shapes[0].NumDims(), 3) << "Wrong rank for " << op;
        }
        
        graph.RemoveNode(node);
    }
}

TEST(ShapeInference, ElementWiseBinary) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "a";
    a.shape = Shape({2, 3});
    b.name = "b";
    b.shape = Shape({2, 3});
    graph.AddInput(a);
    graph.AddInput(b);
    
    std::vector<std::string> ops = {
        "Add", "Sub", "Mul", "Div", "Pow", "Mod", "Fmod",
        "BitShift", "BitwiseAnd", "BitwiseOr", "BitwiseXor"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->AddInput("a");
        node->AddInput("b");
        node->AddOutput("c");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
    }
}

TEST(ShapeInference, ComparisonOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "a";
    a.shape = Shape({2, 3});
    b.name = "b";
    b.shape = Shape({2, 3});
    graph.AddInput(a);
    graph.AddInput(b);
    
    std::vector<std::string> ops = {
        "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->AddInput("a");
        node->AddInput("b");
        node->AddOutput("c");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
    }
}

TEST(ShapeInference, LogicalOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "a";
    a.shape = Shape({2, 3});
    b.name = "b";
    b.shape = Shape({2, 3});
    graph.AddInput(a);
    graph.AddInput(b);
    
    // Binary logical
    for (const auto& op : {"And", "Or", "Xor"}) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, std::string(op) + "_1");
        node->AddInput("a");
        node->AddInput("b");
        node->AddOutput("c");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
    }
    
    // Unary logical
    if (engine.HasHandler("Not")) {
        auto node = graph.CreateNode("Not", "not_1");
        node->AddInput("a");
        node->AddOutput("c");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
    }
}

// ============================================================================
// Linear Algebra Tests
// ============================================================================

TEST(ShapeInference, MatMul) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({3, 4});
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("MatMul", "matmul1");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, Gemm) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({3, 4});
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("Gemm", "gemm1");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, GemmWithBias) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b, c;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({3, 4});
    c.name = "C";
    c.shape = Shape({4});  // 1D bias
    graph.AddInput(a);
    graph.AddInput(b);
    graph.AddInput(c);
    
    auto node = graph.CreateNode("Gemm", "gemm1");
    node->AddInput("A");
    node->AddInput("B");
    node->AddInput("C");  // bias
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, GemmTransposed) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({3, 2});  // Will be transposed to (2, 3)
    b.name = "B";
    b.shape = Shape({4, 3});  // Will be transposed to (3, 4)
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("Gemm", "gemm1");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    node->SetAttribute("transA", static_cast<int64_t>(1));
    node->SetAttribute("transB", static_cast<int64_t>(1));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, QGemm) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({3, 4});
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("QGemm", "qgemm1");
    node->SetDomain("com.microsoft");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, QGemmTransposed) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({3, 2});
    b.name = "B";
    b.shape = Shape({4, 3});
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("QGemm", "qgemm1");
    node->SetDomain("com.microsoft");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    node->SetAttribute("transA", static_cast<int64_t>(1));
    node->SetAttribute("transB", static_cast<int64_t>(1));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

// ============================================================================
// Microsoft Domain Operators Tests
// ============================================================================

TEST(ShapeInference, MicrosoftDomain_QLinearOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3, 4});
    b.name = "B";
    b.shape = Shape({2, 3, 4});
    graph.AddInput(a);
    graph.AddInput(b);
    
    std::vector<std::string> ops = {
        "QLinearAdd", "QLinearMul"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op, "com.microsoft")) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->SetDomain("com.microsoft");
        node->AddInput("A");
        node->AddInput("B");
        node->AddOutput("C");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        
        graph.RemoveNode(node);
    }
}

TEST(ShapeInference, MicrosoftDomain_QLinearPool) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({1, 3, 32, 32});
    graph.AddInput(input);
    
    std::vector<std::string> ops = {
        "QLinearAveragePool", "QLinearGlobalAveragePool"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op, "com.microsoft")) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->SetDomain("com.microsoft");
        node->AddInput("X");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        
        graph.RemoveNode(node);
    }
}

TEST(ShapeInference, MicrosoftDomain_Activations) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    
    std::vector<std::string> ops = {
        "Gelu", "FastGelu"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op, "com.microsoft")) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->SetDomain("com.microsoft");
        node->AddInput("X");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        
        graph.RemoveNode(node);
    }
}

TEST(ShapeInference, MicrosoftDomain_Normalization) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input, skip, gamma, beta;
    input.name = "X";
    input.shape = Shape({2, 3, 256});
    skip.name = "Skip";
    skip.shape = Shape({2, 3, 256});
    gamma.name = "Gamma";
    gamma.shape = Shape({256});
    beta.name = "Beta";
    beta.shape = Shape({256});
    graph.AddInput(input);
    graph.AddInput(skip);
    graph.AddInput(gamma);
    graph.AddInput(beta);
    
    // Test RmsNorm
    if (engine.HasHandler("RmsNorm", "com.microsoft")) {
        auto node = graph.CreateNode("RmsNorm", "rmsnorm1");
        node->SetDomain("com.microsoft");
        node->AddInput("X");
        node->AddInput("Gamma");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
    }
    
    // Test SkipLayerNormalization
    if (engine.HasHandler("SkipLayerNormalization", "com.microsoft")) {
        auto node = graph.CreateNode("SkipLayerNormalization", "skip_ln1");
        node->SetDomain("com.microsoft");
        node->AddInput("X");
        node->AddInput("Skip");
        node->AddInput("Gamma");
        node->AddInput("Beta");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
    }
}

TEST(ShapeInference, MicrosoftDomain_FusedOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({3, 4});
    graph.AddInput(a);
    graph.AddInput(b);
    
    // Test FusedMatMul
    if (engine.HasHandler("FusedMatMul", "com.microsoft")) {
        auto node = graph.CreateNode("FusedMatMul", "fused_matmul1");
        node->SetDomain("com.microsoft");
        node->AddInput("A");
        node->AddInput("B");
        node->AddOutput("C");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    }
    
    // Test FusedGemm
    if (engine.HasHandler("FusedGemm", "com.microsoft")) {
        auto node = graph.CreateNode("FusedGemm", "fused_gemm1");
        node->SetDomain("com.microsoft");
        node->AddInput("A");
        node->AddInput("B");
        node->AddOutput("C");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    }
}

TEST(ShapeInference, MicrosoftDomain_ScatterOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo data, indices, updates;
    data.name = "Data";
    data.shape = Shape({4, 5, 6});
    indices.name = "Indices";
    indices.shape = Shape({2, 3});
    updates.name = "Updates";
    updates.shape = Shape({2, 3, 6});
    graph.AddInput(data);
    graph.AddInput(indices);
    graph.AddInput(updates);
    
    std::vector<std::string> ops = {"Scatter", "ScatterElements", "ScatterND"};
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op, "com.microsoft")) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->SetDomain("com.microsoft");
        node->AddInput("Data");
        node->AddInput("Indices");
        node->AddInput("Updates");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
        
        graph.RemoveNode(node);
    }
}

TEST(ShapeInference, MicrosoftDomain_Trilu) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({4, 4});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Trilu", "trilu1");
    node->SetDomain("com.microsoft");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 4);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, MicrosoftDomain_IsAllFinite) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("IsAllFinite", "is_all_finite1");
    node->SetDomain("com.microsoft");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    // Output is scalar
    EXPECT_EQ(result.output_shapes[0].NumDims(), 0);
}

TEST(ShapeInference, MicrosoftDomain_GridSample) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input, grid;
    input.name = "X";
    input.shape = Shape({1, 3, 32, 32});
    grid.name = "Grid";
    grid.shape = Shape({1, 16, 16, 2});
    graph.AddInput(input);
    graph.AddInput(grid);
    
    auto node = graph.CreateNode("GridSample", "grid_sample1");
    node->SetDomain("com.microsoft");
    node->AddInput("X");
    node->AddInput("Grid");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 4);
}

TEST(ShapeInference, MicrosoftDomain_Attention) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "Input";
    input.shape = Shape({2, 10, 256});  // [batch, seq_len, hidden_size]
    graph.AddInput(input);
    
    if (engine.HasHandler("Attention", "com.microsoft")) {
        auto node = graph.CreateNode("Attention", "attention1");
        node->SetDomain("com.microsoft");
        node->AddInput("Input");
        node->AddOutput("Output");
        node->AddOutput("Present");
        node->AddOutput("QKMatmul");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_GE(result.output_shapes.size(), 1);
    }
    
    if (engine.HasHandler("MultiHeadAttention", "com.microsoft")) {
        ValueInfo q, k, v;
        q.name = "Q"; q.shape = Shape({2, 10, 256});
        k.name = "K"; k.shape = Shape({2, 10, 256});
        v.name = "V"; v.shape = Shape({2, 10, 256});
        graph.AddInput(q);
        graph.AddInput(k);
        graph.AddInput(v);
        
        auto node = graph.CreateNode("MultiHeadAttention", "mha1");
        node->SetDomain("com.microsoft");
        node->AddInput("Q");
        node->AddInput("K");
        node->AddInput("V");
        node->AddOutput("Output");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
    }
}

TEST(ShapeInference, MicrosoftDomain_GradientOps) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input, grad;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    grad.name = "Grad";
    grad.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    graph.AddInput(grad);
    
    std::vector<std::string> ops = {"SinGrad", "CosGrad"};
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op, "com.microsoft")) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->SetDomain("com.microsoft");
        node->AddInput("X");
        node->AddInput("Grad");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
        
        graph.RemoveNode(node);
    }
}

// ============================================================================
// Convolution Tests
// ============================================================================

TEST(ShapeInference, Conv2D) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    
    ValueInfo x, w;
    x.name = "X";
    x.shape = Shape({1, 3, 224, 224});
    w.name = "W";
    w.shape = Shape({64, 3, 7, 7});
    graph.AddInput(x);
    graph.AddInput(w);
    
    auto node = graph.CreateNode("Conv", "conv1");
    node->AddInput("X");
    node->AddInput("W");
    node->AddOutput("Y");
    node->SetAttribute("kernel_shape", std::vector<int64_t>{7, 7});
    node->SetAttribute("strides", std::vector<int64_t>{2, 2});
    node->SetAttribute("pads", std::vector<int64_t>{3, 3, 3, 3});
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 4);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 64);
    EXPECT_EQ(result.output_shapes[0].GetDim(2).GetStaticValue(), 112);
    EXPECT_EQ(result.output_shapes[0].GetDim(3).GetStaticValue(), 112);
}

// ============================================================================
// Shape Operations Tests
// ============================================================================

TEST(ShapeInference, Reshape) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Reshape", "reshape1");
    node->AddInput("X");
    node->AddOutput("Y");
    node->SetAttribute("shape", std::vector<int64_t>{6, 4});
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 6);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 4);
}

TEST(ShapeInference, Transpose) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({1, 3, 224, 224});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Transpose", "transpose1");
    node->AddInput("X");
    node->AddOutput("Y");
    node->SetAttribute("perm", std::vector<int64_t>{0, 2, 3, 1});
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 224);
    EXPECT_EQ(result.output_shapes[0].GetDim(2).GetStaticValue(), 224);
    EXPECT_EQ(result.output_shapes[0].GetDim(3).GetStaticValue(), 3);
}

TEST(ShapeInference, SqueezeUnsqueeze) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({1, 3, 1, 224, 1});
    graph.AddInput(input);
    
    // Test Squeeze
    {
        auto node = graph.CreateNode("Squeeze", "squeeze1");
        node->AddInput("X");
        node->AddOutput("Y");
        node->SetAttribute("axes", std::vector<int64_t>{0, 2, 4});
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
        EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 3);
        EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 224);
    }
    
    // Test Unsqueeze
    {
        auto node = graph.CreateNode("Unsqueeze", "unsqueeze1");
        node->AddInput("X");
        node->AddOutput("Z");
        node->SetAttribute("axes", std::vector<int64_t>{0, 6});
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.output_shapes[0].NumDims(), 7);
    }
}

TEST(ShapeInference, Flatten) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4, 5});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Flatten", "flatten1");
    node->AddInput("X");
    node->AddOutput("Y");
    node->SetAttribute("axis", static_cast<int64_t>(2));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 6);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 20);
}

TEST(ShapeInference, Concat) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo a, b;
    a.name = "A";
    a.shape = Shape({2, 3});
    b.name = "B";
    b.shape = Shape({2, 5});
    graph.AddInput(a);
    graph.AddInput(b);
    
    auto node = graph.CreateNode("Concat", "concat1");
    node->AddInput("A");
    node->AddInput("B");
    node->AddOutput("C");
    node->SetAttribute("axis", static_cast<int64_t>(1));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 2);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 8);
}

TEST(ShapeInference, Gather) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo data, indices;
    data.name = "data";
    data.shape = Shape({10, 20, 30});
    indices.name = "indices";
    indices.shape = Shape({5, 6});
    graph.AddInput(data);
    graph.AddInput(indices);
    
    auto node = graph.CreateNode("Gather", "gather1");
    node->AddInput("data");
    node->AddInput("indices");
    node->AddOutput("output");
    node->SetAttribute("axis", static_cast<int64_t>(1));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 4);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 10);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 5);
    EXPECT_EQ(result.output_shapes[0].GetDim(2).GetStaticValue(), 6);
    EXPECT_EQ(result.output_shapes[0].GetDim(3).GetStaticValue(), 30);
}

TEST(ShapeInference, DepthToSpace) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({1, 12, 8, 8});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("DepthToSpace", "d2s1");
    node->AddInput("X");
    node->AddOutput("Y");
    node->SetAttribute("blocksize", static_cast<int64_t>(2));
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 3);
    EXPECT_EQ(result.output_shapes[0].GetDim(2).GetStaticValue(), 16);
    EXPECT_EQ(result.output_shapes[0].GetDim(3).GetStaticValue(), 16);
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST(ShapeInference, BatchNormalization) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 64, 32, 32});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("BatchNormalization", "bn1");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 4);
}

// ============================================================================
// Activation Tests
// ============================================================================

TEST(ShapeInference, Activations) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({1, 64, 32, 32});
    graph.AddInput(input);
    
    std::vector<std::string> ops = {
        "Relu", "LeakyRelu", "PRelu", "Elu", "Celu", "ThresholdedRelu",
        "Selu", "HardSigmoid", "Shrink", "Softplus", "Softsign", "Clip",
        "HardSwish", "Mish", "Gelu", "QuickGelu", "Swish",
        "Softmax", "LogSoftmax", "Hardmax"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->AddInput("X");
        node->AddOutput("Y");
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        
        graph.RemoveNode(node);
    }
}

// ============================================================================
// Reduction Tests
// ============================================================================

TEST(ShapeInference, Reductions) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4, 5});
    graph.AddInput(input);
    
    std::vector<std::string> ops = {
        "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
        "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
        "ReduceL1", "ReduceL2", "ArgMax", "ArgMin"
    };
    
    for (const auto& op : ops) {
        if (!engine.HasHandler(op)) continue;
        
        auto node = graph.CreateNode(op, op + "_1");
        node->AddInput("X");
        node->AddOutput("Y");
        node->SetAttribute("axes", std::vector<int64_t>{1, 2});
        node->SetAttribute("keepdims", static_cast<int64_t>(1));
        
        auto result = engine.InferNode(node, graph);
        EXPECT_TRUE(result.success) << "Failed for " << op;
        
        graph.RemoveNode(node);
    }
}

// ============================================================================
// Utility Tests
// ============================================================================

TEST(ShapeInference, ShapeOp) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4, 5});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Shape", "shape1");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 4);
}

TEST(ShapeInference, Where) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo cond, x, y;
    cond.name = "cond";
    cond.shape = Shape({2, 3});
    x.name = "x";
    x.shape = Shape({2, 3});
    y.name = "y";
    y.shape = Shape({2, 3});
    graph.AddInput(cond);
    graph.AddInput(x);
    graph.AddInput(y);
    
    auto node = graph.CreateNode("Where", "where1");
    node->AddInput("cond");
    node->AddInput("x");
    node->AddInput("y");
    node->AddOutput("z");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 2);
}

TEST(ShapeInference, Cast) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Cast", "cast1");
    node->AddInput("X");
    node->AddOutput("Y");
    node->SetAttribute("to", static_cast<int64_t>(1));  // FLOAT
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
}

// ============================================================================
// Quantization Tests
// ============================================================================

TEST(ShapeInference, QuantizeLinear) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({2, 3, 4});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("QuantizeLinear", "quant1");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].NumDims(), 3);
}

// ============================================================================
// Dynamic Shape Tests
// ============================================================================

TEST(ShapeInference, DynamicShape) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape(std::vector<Dimension>{Dimension("batch"), Dimension(3), Dimension(224), Dimension(224)});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Relu", "relu1");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_TRUE(result.output_shapes[0].GetDim(0).IsDynamic());
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 3);
}

// ============================================================================
// Full Graph Inference Test
// ============================================================================

TEST(ShapeInference, FullGraph) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    input.shape = Shape({1, 3, 32, 32});
    input.dtype = DataType::kFloat32;
    graph.AddInput(input);
    
    ValueInfo weight;
    weight.name = "weight";
    weight.shape = Shape({16, 3, 3, 3});
    weight.dtype = DataType::kFloat32;
    graph.SetValueInfo("weight", weight);
    
    auto conv = graph.CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    conv->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    conv->SetAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    
    auto relu = graph.CreateNode("Relu", "relu1");
    relu->AddInput("conv_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    output.dtype = DataType::kFloat32;
    graph.AddOutput(output);
    
    ValueInfo conv_out;
    conv_out.name = "conv_out";
    graph.SetValueInfo("conv_out", conv_out);
    
    auto graph_ptr = std::make_shared<Graph>(graph);
    bool success = engine.InferGraph(graph_ptr, false);
    
    EXPECT_TRUE(success);
    
    auto* out_info = graph_ptr->GetValueInfo("conv_out");
    ASSERT_NE(out_info, nullptr);
    EXPECT_EQ(out_info->shape.GetDim(0).GetStaticValue(), 1);
    EXPECT_EQ(out_info->shape.GetDim(1).GetStaticValue(), 16);
    EXPECT_EQ(out_info->shape.GetDim(2).GetStaticValue(), 32);
    EXPECT_EQ(out_info->shape.GetDim(3).GetStaticValue(), 32);
}

// ============================================================================
// Custom Handler Test
// ============================================================================

TEST(ShapeInference, CustomHandler) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    
    engine.Register("CustomOp", [](const InferenceContext& ctx) {
        Shape out = ctx.input_shapes[0];
        if (!out.GetDims().empty() && !out.GetDim(0).IsDynamic()) {
            int64_t new_dim = out.GetDim(0).GetStaticValue() * 2;
            out.GetDim(0) = Dimension(new_dim);
        }
        return InferenceResult::Success({out});
    });
    
    EXPECT_TRUE(engine.HasHandler("CustomOp"));
    
    Graph graph;
    ValueInfo input;
    input.name = "X";
    input.shape = Shape({4, 5});
    graph.AddInput(input);
    
    auto node = graph.CreateNode("CustomOp", "custom1");
    node->AddInput("X");
    node->AddOutput("Y");
    
    auto result = engine.InferNode(node, graph);
    
    EXPECT_TRUE(result.success);
    ASSERT_EQ(result.output_shapes.size(), 1);
    EXPECT_EQ(result.output_shapes[0].GetDim(0).GetStaticValue(), 8);
    EXPECT_EQ(result.output_shapes[0].GetDim(1).GetStaticValue(), 5);
    
    engine.Unregister("CustomOp");
}

// ============================================================================
// Supported Ops Count Test
// ============================================================================

TEST(ShapeInference, SupportedOpsCount) {
    auto& engine = ShapeInferenceEngine::GetInstance();
    auto ops = engine.GetSupportedOps();
    
    std::cout << "Total supported operators: " << ops.size() << std::endl;
    
    // Print some Microsoft domain ops
    int ms_ops = 0;
    for (const auto& op : ops) {
        if (op.find("com.microsoft::") == 0) {
            ms_ops++;
        }
    }
    std::cout << "Microsoft domain operators: " << ms_ops << std::endl;
    
    // Should have at least 150 operators (130 standard + 20+ Microsoft)
    EXPECT_GE(ops.size(), 150u);
}
