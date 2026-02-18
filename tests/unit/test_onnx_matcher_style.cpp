/**
 * @file test_onnx_matcher_style.cpp
 * @brief Unit tests for ONNX Matcher style pattern matching
 */

#include "passes/onnx_matcher_style.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace oniris;
using namespace oniris::passes;

// ============================================================================
// Pattern Parsing Tests
// ============================================================================

TEST(OnnxMatcherPattern, ParseSimpleNode) {
    auto pattern = OnnxMatcherPattern::FromString("Conv(?, c0)");
    ASSERT_TRUE(pattern.has_value());
    ASSERT_EQ(pattern->nodes.size(), 1);
    
    const auto& node = pattern->nodes[0];
    ASSERT_EQ(node.op_types.size(), 1);
    EXPECT_EQ(node.op_types[0], "Conv");
    
    ASSERT_EQ(node.inputs.size(), 1);
    EXPECT_TRUE(node.inputs[0].is_wildcard);
    EXPECT_EQ(node.inputs[0].name, "?");
    
    ASSERT_EQ(node.outputs.size(), 1);
    EXPECT_FALSE(node.outputs[0].is_wildcard);
    EXPECT_EQ(node.outputs[0].name, "c0");
}

TEST(OnnxMatcherPattern, ParseChain) {
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    ASSERT_TRUE(pattern.has_value());
    ASSERT_EQ(pattern->nodes.size(), 2);
    
    // First node
    EXPECT_EQ(pattern->nodes[0].op_types[0], "Conv");
    
    // Second node
    EXPECT_EQ(pattern->nodes[1].op_types[0], "Relu");
    ASSERT_EQ(pattern->nodes[1].inputs.size(), 1);
    EXPECT_EQ(pattern->nodes[1].inputs[0].name, "c0");
}

TEST(OnnxMatcherPattern, ParseMultiOpType) {
    auto pattern = OnnxMatcherPattern::FromString("Conv/MaxPool(?, c0)");
    ASSERT_TRUE(pattern.has_value());
    
    const auto& node = pattern->nodes[0];
    ASSERT_EQ(node.op_types.size(), 2);
    EXPECT_EQ(node.op_types[0], "Conv");
    EXPECT_EQ(node.op_types[1], "MaxPool");
}

TEST(OnnxMatcherPattern, ParseWildcardOpType) {
    auto pattern = OnnxMatcherPattern::FromString("?(?, c0)");
    ASSERT_TRUE(pattern.has_value());
    
    const auto& node = pattern->nodes[0];
    ASSERT_EQ(node.op_types.size(), 1);
    EXPECT_EQ(node.op_types[0], "?");
    EXPECT_TRUE(node.MatchesOpType("Relu"));
    EXPECT_TRUE(node.MatchesOpType("Conv"));
}

TEST(OnnxMatcherPattern, ParseMultiInput) {
    auto pattern = OnnxMatcherPattern::FromString("Mul([s0, c0], ?)");
    ASSERT_TRUE(pattern.has_value());
    
    const auto& node = pattern->nodes[0];
    ASSERT_EQ(node.inputs.size(), 2);
    EXPECT_EQ(node.inputs[0].name, "s0");
    EXPECT_EQ(node.inputs[1].name, "c0");
}

TEST(OnnxMatcherPattern, ParseWithComments) {
    auto pattern = OnnxMatcherPattern::FromString(R"(
        # Match Conv -> ReLU pattern
        Conv(?, c0)
        Relu(c0, ?)  # ReLU takes Conv output
    )");
    
    ASSERT_TRUE(pattern.has_value());
    EXPECT_EQ(pattern->nodes.size(), 2);
}

TEST(OnnxMatcherPattern, ParseEmptyString) {
    auto pattern = OnnxMatcherPattern::FromString("");
    EXPECT_FALSE(pattern.has_value());
}

TEST(OnnxMatcherPattern, ParseOnlyComments) {
    auto pattern = OnnxMatcherPattern::FromString(R"(
        # Only comments
        # No actual pattern
    )");
    EXPECT_FALSE(pattern.has_value());
}

TEST(OnnxMatcherPattern, ParseSwishPattern) {
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Sigmoid(c0, s0)
        Mul([s0, c0], ?)
    )");
    
    ASSERT_TRUE(pattern.has_value());
    ASSERT_EQ(pattern->nodes.size(), 3);
    
    // Check first node: Conv
    EXPECT_EQ(pattern->nodes[0].op_types[0], "Conv");
    
    // Check second node: Sigmoid takes c0, outputs s0
    EXPECT_EQ(pattern->nodes[1].op_types[0], "Sigmoid");
    EXPECT_EQ(pattern->nodes[1].inputs[0].name, "c0");
    EXPECT_EQ(pattern->nodes[1].outputs[0].name, "s0");
    
    // Check third node: Mul takes [s0, c0]
    EXPECT_EQ(pattern->nodes[2].op_types[0], "Mul");
    ASSERT_EQ(pattern->nodes[2].inputs.size(), 2);
    EXPECT_EQ(pattern->nodes[2].inputs[0].name, "s0");
    EXPECT_EQ(pattern->nodes[2].inputs[1].name, "c0");
}

// ============================================================================
// TensorRef Tests
// ============================================================================

TEST(TensorRef, Wildcard) {
    TensorRef ref = TensorRef::Wildcard();
    EXPECT_TRUE(ref.is_wildcard);
    EXPECT_EQ(ref.name, "?");
}

TEST(TensorRef, Variable) {
    TensorRef ref = TensorRef::Variable("c0");
    EXPECT_FALSE(ref.is_wildcard);
    EXPECT_EQ(ref.name, "c0");
}

TEST(TensorRef, StaticConstructors) {
    auto wildcard = TensorRef::Wildcard();
    EXPECT_TRUE(wildcard.is_wildcard);
    EXPECT_EQ(wildcard.name, "?");
    
    auto var = TensorRef::Variable("my_var");
    EXPECT_FALSE(var.is_wildcard);
    EXPECT_EQ(var.name, "my_var");
}

// ============================================================================
// Helper Functions for Creating Test Models
// ============================================================================

static std::shared_ptr<Model> CreateConvReluModel() {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Input
    ValueInfo input;
    input.name = "input";
    input.shape = Shape({1, 3, 224, 224});
    input.dtype = DataType::kFloat;
    graph->AddInput(input);
    
    // Conv
    auto conv = graph->CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    // Relu
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("conv_out");
    relu->AddOutput("output");
    
    // Output
    ValueInfo output;
    output.name = "output";
    output.shape = Shape({1, 64, 224, 224});
    output.dtype = DataType::kFloat;
    graph->AddOutput(output);
    
    return model;
}

static std::shared_ptr<Model> CreateConvReluReluModel() {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Input
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    // Conv
    auto conv = graph->CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    // Relu1
    auto relu1 = graph->CreateNode("Relu", "relu1");
    relu1->AddInput("conv_out");
    relu1->AddOutput("relu1_out");
    
    // Relu2
    auto relu2 = graph->CreateNode("Relu", "relu2");
    relu2->AddInput("relu1_out");
    relu2->AddOutput("output");
    
    // Output
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    return model;
}

static std::shared_ptr<Model> CreateSwishModel() {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Input
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    // Conv
    auto conv = graph->CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddInput("weight");
    conv->AddOutput("conv_out");
    
    // Sigmoid
    auto sigmoid = graph->CreateNode("Sigmoid", "sigmoid1");
    sigmoid->AddInput("conv_out");
    sigmoid->AddOutput("sigmoid_out");
    
    // Mul
    auto mul = graph->CreateNode("Mul", "mul1");
    mul->AddInput("sigmoid_out");
    mul->AddInput("conv_out");
    mul->AddOutput("output");
    
    // Output
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    return model;
}

static std::shared_ptr<Model> CreateParallelConvsModel() {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Input
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    // Conv1 -> Relu1
    auto conv1 = graph->CreateNode("Conv", "conv1");
    conv1->AddInput("input");
    conv1->AddInput("weight1");
    conv1->AddOutput("conv1_out");
    
    auto relu1 = graph->CreateNode("Relu", "relu1");
    relu1->AddInput("conv1_out");
    relu1->AddOutput("relu1_out");
    
    // Conv2 -> Relu2
    auto conv2 = graph->CreateNode("Conv", "conv2");
    conv2->AddInput("input");
    conv2->AddInput("weight2");
    conv2->AddOutput("conv2_out");
    
    auto relu2 = graph->CreateNode("Relu", "relu2");
    relu2->AddInput("conv2_out");
    relu2->AddOutput("relu2_out");
    
    // Add
    auto add = graph->CreateNode("Add", "add1");
    add->AddInput("relu1_out");
    add->AddInput("relu2_out");
    add->AddOutput("output");
    
    // Output
    ValueInfo output;
    output.name = "output";
    graph->AddOutput(output);
    
    return model;
}

// ============================================================================
// Matcher Tests - Simple Patterns
// ============================================================================

TEST(OnnxMatcherStyleMatcher, FindAll_ConvRelu) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    ASSERT_TRUE(pattern.has_value());
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);
    
    const auto& match = matches[0];
    ASSERT_EQ(match.matched_nodes.size(), 2);
    EXPECT_EQ(match.matched_nodes[0]->GetOpType(), "Conv");
    EXPECT_EQ(match.matched_nodes[1]->GetOpType(), "Relu");
}

TEST(OnnxMatcherStyleMatcher, FindAll_ConvReluRelu) {
    auto model = CreateConvReluReluModel();
    
    // Pattern: Conv -> Relu
    auto pattern1 = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    auto matches1 = OnnxMatcherStyleMatcher::FindAll(model, *pattern1);
    // Note: Implementation finds complete chains only, so we get 1 match
    EXPECT_EQ(matches1.size(), 1);  // Matches the complete Conv -> Relu chain
    
    // Pattern: Conv -> Relu -> Relu
    auto pattern2 = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, c1)
        Relu(c1, ?)
    )");
    
    auto matches2 = OnnxMatcherStyleMatcher::FindAll(model, *pattern2);
    EXPECT_EQ(matches2.size(), 1);  // Only one complete chain
}

TEST(OnnxMatcherStyleMatcher, HasMatch) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString("Conv(?, ?)");
    
    EXPECT_TRUE(OnnxMatcherStyleMatcher::HasMatch(model, *pattern));
    
    auto nonexistent = OnnxMatcherPattern::FromString("NonExistentOp(?, ?)");
    EXPECT_FALSE(OnnxMatcherStyleMatcher::HasMatch(model, *nonexistent));
}

TEST(OnnxMatcherStyleMatcher, FindFirst) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    auto match = OnnxMatcherStyleMatcher::FindFirst(model, *pattern);
    EXPECT_TRUE(match.IsValid());
    ASSERT_EQ(match.matched_nodes.size(), 2);
}

TEST(OnnxMatcherStyleMatcher, FindFirst_NoMatch) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString("NonExistentOp(?, ?)");
    
    auto match = OnnxMatcherStyleMatcher::FindFirst(model, *pattern);
    EXPECT_FALSE(match.IsValid());
    EXPECT_TRUE(match.matched_nodes.empty());
}

// ============================================================================
// Matcher Tests - Complex Patterns
// ============================================================================

TEST(OnnxMatcherStyleMatcher, SwishPattern) {
    auto model = CreateSwishModel();
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Sigmoid(c0, s0)
        Mul([s0, c0], ?)
    )");
    ASSERT_TRUE(pattern.has_value());
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);
    
    const auto& match = matches[0];
    ASSERT_EQ(match.matched_nodes.size(), 3);
    EXPECT_EQ(match.matched_nodes[0]->GetOpType(), "Conv");
    EXPECT_EQ(match.matched_nodes[1]->GetOpType(), "Sigmoid");
    EXPECT_EQ(match.matched_nodes[2]->GetOpType(), "Mul");
}

TEST(OnnxMatcherStyleMatcher, WildcardOpType) {
    auto model = CreateConvReluModel();
    
    // Match any op followed by Relu
    auto pattern = OnnxMatcherPattern::FromString(R"(
        ?(?, c0)
        Relu(c0, ?)
    )");
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);
}

TEST(OnnxMatcherStyleMatcher, MultiOpType) {
    auto model = CreateConvReluModel();
    
    // Conv/Relu should match Conv
    auto pattern = OnnxMatcherPattern::FromString("Conv/Relu(?, ?)");
    ASSERT_TRUE(pattern.has_value());
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 2);  // Both Conv and Relu match
}

// ============================================================================
// Matcher Tests - Edge Cases
// ============================================================================

TEST(OnnxMatcherStyleMatcher, EmptyModel) {
    auto model = std::make_shared<Model>();
    // Model without graph set
    auto pattern = OnnxMatcherPattern::FromString("Conv(?, ?)");
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_TRUE(matches.empty());
}

TEST(OnnxMatcherStyleMatcher, EmptyPattern) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString("");
    
    EXPECT_FALSE(pattern.has_value());
}

TEST(OnnxMatcherStyleMatcher, SingleNodePattern) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString("Conv(?, ?)");
    ASSERT_TRUE(pattern.has_value());
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0].matched_nodes.size(), 1);
}

TEST(OnnxMatcherStyleMatcher, ParallelConvs) {
    auto model = CreateParallelConvsModel();
    
    // Should find two Conv -> Relu patterns
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 2);
}

TEST(OnnxMatcherStyleMatcher, MatchLongerChain) {
    auto model = CreateConvReluReluModel();
    
    // Pattern matching 3 nodes
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, c1)
        Relu(c1, ?)
    )");
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);  // One complete chain
    EXPECT_EQ(matches[0].matched_nodes.size(), 3);
}

TEST(OnnxMatcherStyleMatcher, NodeMapping) {
    auto model = CreateConvReluModel();
    auto pattern = OnnxMatcherPattern::FromString(R"(
        Conv(?, c0)
        Relu(c0, ?)
    )");
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    ASSERT_EQ(matches.size(), 1);
    
    const auto& match = matches[0];
    EXPECT_FALSE(match.node_mapping.empty());
    EXPECT_TRUE(match.node_mapping.count("node0") > 0);
    EXPECT_TRUE(match.node_mapping.count("node1") > 0);
    EXPECT_EQ(match.node_mapping.at("node0")->GetOpType(), "Conv");
    EXPECT_EQ(match.node_mapping.at("node1")->GetOpType(), "Relu");
}

TEST(OnnxMatcherStyleMatcher, MatchNoConstants) {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Add input
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    // Add a relu
    auto relu = graph->CreateNode("Relu", "relu1");
    relu->AddInput("input");
    relu->AddOutput("output");
    
    // Pattern for single Relu
    auto pattern = OnnxMatcherPattern::FromString("Relu(?, ?)");
    
    // Should find the match
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    EXPECT_EQ(matches.size(), 1);
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

TEST(OnnxMatcherStyleMatcher, LongChain) {
    auto model = std::make_shared<Model>();
    auto graph = std::make_shared<Graph>();
    model->SetGraph(graph);
    
    // Create a chain of 10 Relus
    ValueInfo input;
    input.name = "input";
    graph->AddInput(input);
    
    std::string prev = "input";
    for (int i = 0; i < 10; ++i) {
        auto relu = graph->CreateNode("Relu", "relu" + std::to_string(i));
        relu->AddInput(prev);
        std::string out = "relu" + std::to_string(i) + "_out";
        relu->AddOutput(out);
        prev = out;
    }
    
    ValueInfo output;
    output.name = prev;
    graph->AddOutput(output);
    
    // Pattern for chain of 5 Relus
    std::string pattern_str = R"(
        Relu(?, t0)
        Relu(t0, t1)
        Relu(t1, t2)
        Relu(t2, t3)
        Relu(t3, ?)
    )";
    
    auto pattern = OnnxMatcherPattern::FromString(pattern_str);
    ASSERT_TRUE(pattern.has_value());
    
    auto matches = OnnxMatcherStyleMatcher::FindAll(model, *pattern);
    // Should find multiple overlapping matches
    EXPECT_GE(matches.size(), 1);
}
