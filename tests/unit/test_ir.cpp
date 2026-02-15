/**
 * @file test_ir.cpp
 * @brief Unit tests for Intermediate Representation
 */

#include "ir/graph.hpp"
#include "ir/model.hpp"
#include "ir/node.hpp"
#include "ir/tensor.hpp"

#include <gtest/gtest.h>

using namespace oniris;

// =============================================================================
// Node Tests
// =============================================================================

TEST(NodeTest, CreateNode) {
    auto node = std::make_shared<Node>("Conv", "conv1");
    EXPECT_EQ(node->GetOpType(), "Conv");
    EXPECT_EQ(node->GetName(), "conv1");
}

TEST(NodeTest, InputsOutputs) {
    auto node = std::make_shared<Node>("Add", "add1");
    
    node->AddInput("input_a");
    node->AddInput("input_b");
    node->AddOutput("output");
    
    EXPECT_EQ(node->GetInputs().size(), 2);
    EXPECT_EQ(node->GetOutputs().size(), 1);
    EXPECT_EQ(node->GetInputs()[0], "input_a");
    EXPECT_EQ(node->GetOutputs()[0], "output");
}

TEST(NodeTest, Attributes) {
    auto node = std::make_shared<Node>("Conv", "conv1");
    
    node->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    node->SetAttribute("strides", std::vector<int64_t>{1, 1});
    node->SetAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    
    EXPECT_TRUE(node->HasAttribute("kernel_shape"));
    EXPECT_FALSE(node->HasAttribute("dilations"));
    
    auto kernel = node->GetAttributeAs<std::vector<int64_t>>("kernel_shape");
    EXPECT_TRUE(kernel.has_value());
    EXPECT_EQ(kernel->size(), 2);
    EXPECT_EQ((*kernel)[0], 3);
}

TEST(NodeTest, Clone) {
    auto node = std::make_shared<Node>("Conv", "conv1");
    node->AddInput("x");
    node->AddOutput("y");
    node->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    
    auto cloned = node->Clone();
    
    EXPECT_EQ(cloned->GetOpType(), node->GetOpType());
    EXPECT_EQ(cloned->GetName(), node->GetName());
    EXPECT_EQ(cloned->GetInputs().size(), node->GetInputs().size());
    EXPECT_TRUE(cloned->HasAttribute("kernel_shape"));
}

// =============================================================================
// Tensor Tests
// =============================================================================

TEST(TensorTest, CreateTensor) {
    Shape shape({2, 3, 4});
    Tensor tensor(shape, DataType::kFloat32);
    
    EXPECT_EQ(tensor.GetShape().NumDims(), 3);
    EXPECT_EQ(tensor.GetDataType(), DataType::kFloat32);
    EXPECT_FALSE(tensor.HasData());
}

TEST(TensorTest, CreateWithData) {
    Shape shape({2, 3});
    std::vector<uint8_t> data(24, 0);  // 6 floats * 4 bytes
    Tensor tensor(shape, DataType::kFloat32, data);
    
    EXPECT_TRUE(tensor.HasData());
    EXPECT_EQ(tensor.GetData().size(), 24);
    
    auto num_elems = tensor.GetNumElements();
    EXPECT_TRUE(num_elems.has_value());
    EXPECT_EQ(*num_elems, 6);
}

TEST(TensorTest, ScalarTensor) {
    Tensor tensor(Shape(), DataType::kFloat32);
    
    EXPECT_TRUE(tensor.IsScalar());
    EXPECT_EQ(tensor.GetNumElements(), 1);
}

// =============================================================================
// Graph Tests
// =============================================================================

TEST(GraphTest, CreateGraph) {
    Graph graph("test_graph");
    EXPECT_EQ(graph.GetName(), "test_graph");
    EXPECT_TRUE(graph.GetNodes().empty());
}

TEST(GraphTest, AddNodes) {
    Graph graph;
    
    auto node1 = graph.CreateNode("Conv", "conv1");
    auto node2 = graph.CreateNode("Relu", "relu1");
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    EXPECT_EQ(node1->GetOpType(), "Conv");
    EXPECT_EQ(node2->GetOpType(), "Relu");
}

TEST(GraphTest, InputsOutputs) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    input.shape = Shape({1, 3, 224, 224});
    input.dtype = DataType::kFloat32;
    graph.AddInput(input);
    
    ValueInfo output;
    output.name = "output";
    output.shape = Shape({1, 1000});
    output.dtype = DataType::kFloat32;
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetInputs().size(), 1);
    EXPECT_EQ(graph.GetOutputs().size(), 1);
    EXPECT_EQ(graph.GetInputs()[0].name, "input");
}

TEST(GraphTest, ValueInfo) {
    Graph graph;
    
    ValueInfo info;
    info.name = "intermediate";
    info.shape = Shape({1, 64, 56, 56});
    info.dtype = DataType::kFloat32;
    graph.SetValueInfo("intermediate", info);
    
    EXPECT_TRUE(graph.HasValueInfo("intermediate"));
    EXPECT_FALSE(graph.HasValueInfo("missing"));
    
    auto* retrieved = graph.GetValueInfo("intermediate");
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->name, "intermediate");
}

TEST(GraphTest, Constants) {
    Graph graph;
    
    ConstantTensor ct;
    ct.name = "weight";
    ct.shape = Shape({64, 3, 7, 7});
    ct.dtype = DataType::kFloat32;
    graph.AddConstant("weight", ct);
    
    EXPECT_TRUE(graph.HasConstant("weight"));
    
    auto* retrieved = graph.GetConstant("weight");
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->name, "weight");
}

TEST(GraphTest, ProducerConsumer) {
    Graph graph;
    
    auto node1 = graph.CreateNode("Conv", "conv1");
    node1->AddInput("input");
    node1->AddOutput("mid");
    
    auto node2 = graph.CreateNode("Relu", "relu1");
    node2->AddInput("mid");
    node2->AddOutput("output");
    
    auto producer = graph.GetProducer("mid");
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->GetName(), "conv1");
    
    auto consumers = graph.GetConsumers("mid");
    EXPECT_EQ(consumers.size(), 1);
    EXPECT_EQ(consumers[0]->GetName(), "relu1");
}

TEST(GraphTest, TopologicalSort) {
    Graph graph;
    
    // Create nodes: input -> conv -> relu -> output
    auto relu = graph.CreateNode("Relu", "relu1");
    relu->AddInput("conv_out");
    relu->AddOutput("output");
    
    auto conv = graph.CreateNode("Conv", "conv1");
    conv->AddInput("input");
    conv->AddOutput("conv_out");
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto sorted = graph.TopologicalSort();
    
    ASSERT_EQ(sorted.size(), 2);
    EXPECT_EQ(sorted[0]->GetName(), "conv1");
    EXPECT_EQ(sorted[1]->GetName(), "relu1");
}

TEST(GraphTest, RemoveDeadNodes) {
    Graph graph;
    
    // Create: input -> conv1 -> output
    //              -> conv2 (dead)
    auto conv1 = graph.CreateNode("Conv", "conv1");
    conv1->AddInput("input");
    conv1->AddOutput("output");
    
    auto conv2 = graph.CreateNode("Conv", "conv2");
    conv2->AddInput("input");
    conv2->AddOutput("unused");
    
    ValueInfo input, output;
    input.name = "input";
    output.name = "output";
    graph.AddInput(input);
    graph.AddOutput(output);
    
    EXPECT_EQ(graph.GetNodes().size(), 2);
    
    graph.RemoveDeadNodes();
    
    EXPECT_EQ(graph.GetNodes().size(), 1);
    EXPECT_EQ(graph.GetNodes()[0]->GetName(), "conv1");
}

TEST(GraphTest, Validate) {
    Graph graph;
    
    ValueInfo input;
    input.name = "input";
    graph.AddInput(input);
    
    auto node = graph.CreateNode("Conv", "conv1");
    node->AddInput("input");
    node->AddOutput("output");
    
    ValueInfo output;
    output.name = "output";
    graph.AddOutput(output);
    
    std::string error_msg;
    EXPECT_TRUE(graph.Validate(&error_msg));
    
    // Test invalid graph: undefined input
    node->SetInput(0, "undefined");
    EXPECT_FALSE(graph.Validate(&error_msg));
}

// =============================================================================
// Model Tests
// =============================================================================

TEST(ModelTest, CreateModel) {
    Model model(8);
    
    EXPECT_EQ(model.GetIRVersion(), 8);
    EXPECT_EQ(model.GetProducerName(), "oniris");
}

TEST(ModelTest, OpsetImports) {
    Model model;
    
    OpsetImport opset;
    opset.domain = "";
    opset.version = 13;
    model.AddOpsetImport(opset);
    
    EXPECT_EQ(model.GetOpsetImports().size(), 1);
    EXPECT_EQ(model.GetOpsetImports()[0].version, 13);
}

TEST(ModelTest, Graph) {
    Model model;
    
    auto graph = model.CreateGraph("main");
    EXPECT_NE(graph, nullptr);
    EXPECT_EQ(graph->GetName(), "main");
    EXPECT_EQ(model.GetGraph(), graph);
}

TEST(ModelTest, Validate) {
    Model model;
    
    std::string error_msg;
    EXPECT_FALSE(model.Validate(&error_msg));  // No graph
    
    model.CreateGraph("main");
    EXPECT_FALSE(model.Validate(&error_msg));  // No opset
    
    OpsetImport opset;
    opset.version = 13;
    model.AddOpsetImport(opset);
    EXPECT_TRUE(model.Validate(&error_msg));
}
