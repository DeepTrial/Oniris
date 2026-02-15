/**
 * @file test_types.cpp
 * @brief Unit tests for core types
 */

#include "core/types.hpp"

#include <gtest/gtest.h>

using namespace oniris;

// =============================================================================
// Dimension Tests
// =============================================================================

TEST(DimensionTest, StaticDimension) {
    Dimension d(42);
    EXPECT_FALSE(d.IsDynamic());
    EXPECT_EQ(d.GetStaticValue(), 42);
}

TEST(DimensionTest, DynamicDimension) {
    Dimension d("batch_size");
    EXPECT_TRUE(d.IsDynamic());
    EXPECT_EQ(d.GetSymbolicName(), "batch_size");
}

TEST(DimensionTest, SetStaticValue) {
    Dimension d("batch");
    d.SetStaticValue(10);
    EXPECT_FALSE(d.IsDynamic());
    EXPECT_EQ(d.GetStaticValue(), 10);
}

TEST(DimensionTest, SetDynamic) {
    Dimension d(42);
    d.SetDynamic("seq_len");
    EXPECT_TRUE(d.IsDynamic());
    EXPECT_EQ(d.GetSymbolicName(), "seq_len");
}

TEST(DimensionTest, ToString) {
    Dimension d1(42);
    EXPECT_EQ(d1.ToString(), "42");
    
    Dimension d2;
    EXPECT_EQ(d2.ToString(), "?");
    
    Dimension d3("batch");
    EXPECT_EQ(d3.ToString(), "batch");
}

TEST(DimensionTest, Equality) {
    Dimension d1(42);
    Dimension d2(42);
    Dimension d3(10);
    Dimension d4("batch");
    
    EXPECT_EQ(d1, d2);
    EXPECT_NE(d1, d3);
    EXPECT_NE(d1, d4);
}

// =============================================================================
// Shape Tests
// =============================================================================

TEST(ShapeTest, EmptyShape) {
    Shape s;
    EXPECT_EQ(s.NumDims(), 0);
    EXPECT_TRUE(s.IsScalar());
    EXPECT_FALSE(s.IsDynamic());
}

TEST(ShapeTest, StaticShape) {
    Shape s({1, 3, 224, 224});
    EXPECT_EQ(s.NumDims(), 4);
    EXPECT_FALSE(s.IsScalar());
    EXPECT_FALSE(s.IsDynamic());
    EXPECT_EQ(s.GetDim(0).GetStaticValue(), 1);
    EXPECT_EQ(s.GetDim(1).GetStaticValue(), 3);
    EXPECT_EQ(s.GetDim(2).GetStaticValue(), 224);
    EXPECT_EQ(s.GetDim(3).GetStaticValue(), 224);
}

TEST(ShapeTest, DynamicShape) {
    Shape s;
    s.AddDim(Dimension("batch"));
    s.AddDim(3);
    s.AddDim(224);
    s.AddDim(224);
    
    EXPECT_EQ(s.NumDims(), 4);
    EXPECT_TRUE(s.IsDynamic());
    EXPECT_FALSE(s.IsStatic());
}

TEST(ShapeTest, GetTotalSize) {
    Shape s({2, 3, 4});
    auto total = s.GetTotalSize();
    EXPECT_TRUE(total.has_value());
    EXPECT_EQ(*total, 24);
}

TEST(ShapeTest, GetTotalSizeDynamic) {
    Shape s;
    s.AddDim(Dimension());
    s.AddDim(3);
    
    auto total = s.GetTotalSize();
    EXPECT_FALSE(total.has_value());
}

TEST(ShapeTest, ToString) {
    Shape s1({1, 3, 224, 224});
    EXPECT_EQ(s1.ToString(), "[1, 3, 224, 224]");
    
    Shape s2;
    s2.AddDim(Dimension("batch"));
    s2.AddDim(3);
    EXPECT_EQ(s2.ToString(), "[batch, 3]");
}

// =============================================================================
// DataType Tests
// =============================================================================

TEST(DataTypeTest, ToString) {
    EXPECT_EQ(DataTypeToString(DataType::kFloat32), "float32");
    EXPECT_EQ(DataTypeToString(DataType::kInt64), "int64");
    EXPECT_EQ(DataTypeToString(DataType::kUnknown), "unknown");
}

TEST(DataTypeTest, FromString) {
    EXPECT_EQ(StringToDataType("float32"), DataType::kFloat32);
    EXPECT_EQ(StringToDataType("tensor(float)"), DataType::kFloat32);
    EXPECT_EQ(StringToDataType("int64"), DataType::kInt64);
    EXPECT_EQ(StringToDataType("unknown_op"), DataType::kUnknown);
}

TEST(DataTypeTest, GetSize) {
    EXPECT_EQ(GetDataTypeSize(DataType::kFloat32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt64), 8);
    EXPECT_EQ(GetDataTypeSize(DataType::kUint8), 1);
    EXPECT_EQ(GetDataTypeSize(DataType::kString), 0);
}
