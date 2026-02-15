/**
 * @file test_simplifier.cpp
 * @brief Unit tests for model simplifier (including fusion)
 */

#include "passes/simplifier.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace oniris;
using namespace oniris::passes;

// ============================================================================
// Identity and Nop Elimination Tests
// ============================================================================

TEST(Simplifier, EliminateIdentity) {
    auto graph = std::make_shared<Graph>();
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto identity = graph->CreateNode("Identity", "id1");
    identity->AddInput("input");
    identity->AddOutput("id_out");
    
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("id_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    EXPECT_EQ(graph->GetNodes().size(), 2);
    
    SimplifyOptions options;
    options.skip_shape_inference = true;
    options.skip_constant_folding = true;
    
    auto result = Simplifier::SimplifyGraph(graph, options);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.num_changes, 1);
    EXPECT_EQ(graph->GetNodes().size(), 1);
    EXPECT_EQ(graph->GetNodes()[0]->GetInputs()[0], "input");
}

TEST(Simplifier, EliminateNopTranspose) {
    auto graph = std::make_shared<Graph>();
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto transpose = graph->CreateNode("Transpose", "transpose1");
    transpose->AddInput("input");
    transpose->AddOutput("t_out");
    transpose->SetAttribute("perm", std::vector<int64_t>{0, 1, 2});
    
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("t_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    EXPECT_EQ(graph->GetNodes().size(), 2);
    
    SimplifyOptions options;
    options.skip_shape_inference = true;
    options.skip_constant_folding = true;
    
    auto result = Simplifier::SimplifyGraph(graph, options);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.num_changes, 1);
    EXPECT_EQ(graph->GetNodes().size(), 1);
}

TEST(Simplifier, EliminateDeadNodes) {
    auto graph = std::make_shared<Graph>();
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto relu1 = graph->CreateNode("Relu", "relu1");
    relu1->AddInput("input");
    relu1->AddOutput("mid");
    
    auto relu2 = graph->CreateNode("Relu", "relu2");
    relu2->AddInput("input");
    relu2->AddOutput("dead");
    
    auto relu3 = graph->CreateNode("Relu", "relu3");
    relu3->AddInput("mid");
    relu3->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    EXPECT_EQ(graph->GetNodes().size(), 3);
    
    SimplifyOptions options;
    options.skip_shape_inference = true;
    options.skip_constant_folding = true;
    
    auto result = Simplifier::SimplifyGraph(graph, options);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(graph->GetNodes().size(), 2);
}

// ============================================================================
// Fusion Tests
// ============================================================================

TEST(Simplifier, FuseConvBN) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    input.shape = Shape({1, 3, 32, 32});
    graph.AddInput(input);
    
    // Setup BN constants
    ConstantTensor ct_scale, ct_bias, ct_mean, ct_var;
    ct_scale.name = "bn_scale";
    ct_scale.shape = Shape({16});
    ct_scale.dtype = DataType::kFloat32;
    ct_bias.name = "bn_bias";
    ct_bias.shape = Shape({16});
    ct_bias.dtype = DataType::kFloat32;
    ct_mean.name = "bn_mean";
    ct_mean.shape = Shape({16});
    ct_mean.dtype = DataType::kFloat32;
    ct_var.name = "bn_var";
    ct_var.shape = Shape({16});
    ct_var.dtype = DataType::kFloat32;
    
    graph.AddConstant("bn_scale", ct_scale);
    graph.AddConstant("bn_bias", ct_bias);
    graph.AddConstant("bn_mean", ct_mean);
    graph.AddConstant("bn_var", ct_var);
    
    auto conv = graph.CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    auto bn = graph.CreateNode("BatchNormalization", "bn1");
    bn->AddInput("conv_out");
    bn->AddInput("bn_scale");
    bn->AddInput("bn_bias");
    bn->AddInput("bn_mean");
    bn->AddInput("bn_var");
    bn->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    
    int fused = Simplifier::FuseConvBN(graph);
    
    EXPECT_EQ(fused, 1);
    EXPECT_EQ(graph.GetNodes().size(), 1);
    EXPECT_EQ(graph.GetNodes()[0]->GetOpType(), "Conv");
}

TEST(Simplifier, FuseConvRelu) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto conv = graph.CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    auto relu = graph.CreateNode("Relu", "relu1");
    relu->AddInput("conv_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    
    int fused = Simplifier::FuseConvRelu(graph);
    
    EXPECT_EQ(fused, 1);
    EXPECT_EQ(graph.GetNodes().size(), 1);
}

TEST(Simplifier, FuseGemmActivation) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto gemm = graph.CreateNode("Gemm", "gemm1");
    gemm->AddInput("input");
    gemm->AddInput("weight");
    gemm->AddOutput("gemm_out");
    
    auto relu = graph.CreateNode("Relu", "relu1");
    relu->AddInput("gemm_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    
    int fused = Simplifier::FuseGemmActivations(graph);
    
    EXPECT_EQ(fused, 1);
    EXPECT_EQ(graph.GetNodes().size(), 1);
}

TEST(Simplifier, FusionDisabled) {
    auto graph = std::make_shared<Graph>();
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto conv = graph->CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("conv_out");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    EXPECT_EQ(graph->GetNodes().size(), 2);
    
    SimplifyOptions options;
    options.fuse_conv_relu = false;  // Disable fusion
    options.skip_shape_inference = true;
    
    auto result = Simplifier::SimplifyGraph(graph, options);
    
    // Should not fuse, but may eliminate other nops
    EXPECT_EQ(graph->GetNodes().size(), 2);
}

// ============================================================================
// Nop Pad/Slice/Resize Tests
// ============================================================================

TEST(Simplifier, EliminateNopPad) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto pad = graph.CreateNode("Pad", "pad1");
    pad->AddInput("input");
    pad->AddOutput("output");
    pad->SetAttribute("pads", std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0});
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 1);
    
    int eliminated = Simplifier::EliminateNopPad(graph);
    
    EXPECT_EQ(eliminated, 1);
    EXPECT_EQ(graph.GetNodes().size(), 0);
}

TEST(Simplifier, EliminateNopResize) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto resize = graph.CreateNode("Resize", "resize1");
    resize->AddInput("input");
    resize->AddInput("scales");
    resize->AddOutput("output");
    resize->SetAttribute("scales", std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 1);
    
    int eliminated = Simplifier::EliminateNopResize(graph);
    
    EXPECT_EQ(eliminated, 1);
    EXPECT_EQ(graph.GetNodes().size(), 0);
}

// ============================================================================
// Model Level Simplification
// ============================================================================

TEST(Simplifier, SimplifyModel) {
    auto model = std::make_shared<Model>(8);
    
    OpsetImport opset;
    opset.version = 13;
    model->AddOpsetImport(opset);
    
    auto graph = model->CreateGraph("test");
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto identity = graph->CreateNode("Identity", "id1");
    identity->AddInput("input");
    identity->AddOutput("mid");
    
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("mid");
    relu->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    EXPECT_EQ(graph->GetNodes().size(), 2);
    
    SimplifyOptions options;
    options.skip_shape_inference = true;
    
    auto result = Simplifier::Simplify(model, options);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.num_changes, 1);
    EXPECT_EQ(graph->GetNodes().size(), 1);
}

TEST(Simplifier, EmptyModel) {
    auto model = std::make_shared<Model>(8);
    
    SimplifyOptions options;
    auto result = Simplifier::Simplify(model, options);
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_msg.empty());
}

TEST(Simplifier, NullModel) {
    SimplifyOptions options;
    auto result = Simplifier::Simplify(nullptr, options);
    
    EXPECT_FALSE(result.success);
}

// ============================================================================
// RunAllPasses Test
// ============================================================================

TEST(Simplifier, RunAllPasses) {
    Graph graph;
    
    // Create various nop operations
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    // Nop transpose
    auto transpose = graph.CreateNode("Transpose", "transpose1");
    transpose->AddInput("input");
    transpose->AddOutput("t_out");
    transpose->SetAttribute("perm", std::vector<int64_t>{0, 1, 2});
    
    // Single input concat
    auto concat = graph.CreateNode("Concat", "concat1");
    concat->AddInput("t_out");
    concat->AddOutput("output");
    concat->SetAttribute("axis", static_cast<int64_t>(0));
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    
    SimplifyOptions options;
    options.skip_shape_inference = true;
    options.skip_constant_folding = true;
    
    int changes = Simplifier::RunAllPasses(graph, options);
    
    EXPECT_GT(changes, 0);
}

// ============================================================================
// Options Control Test
// ============================================================================

TEST(Simplifier, OptionsControl) {
    auto graph = std::make_shared<Graph>();
    
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    auto identity = graph->CreateNode("Identity", "id1");
    identity->AddInput("input");
    identity->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    // Test with identity elimination disabled
    SimplifyOptions options;
    options.skip_identity_elimination = true;
    options.skip_shape_inference = true;
    
    auto result = Simplifier::SimplifyGraph(graph, options);
    
    // Identity should still be there
    EXPECT_EQ(graph->GetNodes().size(), 1);
    
    // Test with identity elimination enabled
    options.skip_identity_elimination = false;
    result = Simplifier::SimplifyGraph(graph, options);
    
    // Identity should be eliminated now
    EXPECT_EQ(graph->GetNodes().size(), 0);
}
