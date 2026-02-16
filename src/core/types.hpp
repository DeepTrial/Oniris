/**
 * @file types.hpp
 * @brief Core type definitions for Oniris
 * 
 * AUTO-GENERATED from ONNX proto file (IR_VERSION: 0x000000000000000D)
 * Do not edit manually. Run scripts/generate_from_proto.py to regenerate.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oniris {

/**
 * @brief Data type enumeration matching ONNX TensorProto.DataType
 * IR_VERSION: 0x000000000000000D
 */
enum class DataType : int {
    // ONNX standard types (auto-generated from onnx.proto)
    kUndefined = 0,
    kFloat = 1,
    kUint8 = 2,
    kInt8 = 3,
    kUint16 = 4,
    kInt16 = 5,
    kInt32 = 6,
    kInt64 = 7,
    kString = 8,
    kBool = 9,
    kFloat16 = 10,
    kDouble = 11,
    kUint32 = 12,
    kUint64 = 13,
    kComplex64 = 14,
    kComplex128 = 15,
    kBFloat16 = 16,
    kFloat8E4M3FN = 17,
    kFloat8E4M3FNUZ = 18,
    kFloat8E5M2 = 19,
    kFloat8E5M2FNUZ = 20,
    kUint4 = 21,
    kInt4 = 22,
    kFloat4E2M1 = 23,
    kFloat8E8M0 = 24,
    kUint2 = 25,
    kInt2 = 26,
    
    // Aliases for backward compatibility
    kUnknown = kUndefined,
    kFloat32 = kFloat,
    kFloat64 = kDouble,
};

std::string DataTypeToString(DataType dtype);
DataType StringToDataType(const std::string& str);
size_t GetDataTypeSize(DataType dtype);

class Dimension {
public:
    explicit Dimension(int64_t value) : symbol_(""), value_(value), is_dynamic_(false) {}
    explicit Dimension(const std::string& symbol = "")
        : symbol_(symbol), value_(-1), is_dynamic_(true) {}
    bool IsDynamic() const { return is_dynamic_; }
    int64_t GetStaticValue() const { return value_; }
    const std::string& GetSymbolicName() const { return symbol_; }
    void SetStaticValue(int64_t value) {
        value_ = value;
        is_dynamic_ = false;
        symbol_.clear();
    }
    void SetDynamic(const std::string& symbol = "") {
        symbol_ = symbol;
        is_dynamic_ = true;
        value_ = -1;
    }
    std::string ToString() const;
    bool operator==(const Dimension& other) const;
    bool operator!=(const Dimension& other) const { return !(*this == other); }

private:
    std::string symbol_;
    int64_t value_;
    bool is_dynamic_;
};

class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims);
    explicit Shape(std::vector<Dimension> dims) : dims_(std::move(dims)) {}
    size_t NumDims() const { return dims_.size(); }
    bool IsScalar() const { return dims_.empty(); }
    bool IsDynamic() const;
    bool IsStatic() const { return !IsDynamic(); }
    const Dimension& GetDim(size_t idx) const { return dims_.at(idx); }
    Dimension& GetDim(size_t idx) { return dims_.at(idx); }
    const std::vector<Dimension>& GetDims() const { return dims_; }
    std::vector<Dimension>& GetDims() { return dims_; }
    void AddDim(const Dimension& dim) { dims_.push_back(dim); }
    void AddDim(int64_t value) { dims_.emplace_back(value); }
    std::optional<int64_t> GetTotalSize() const;
    std::string ToString() const;
    bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    std::vector<Dimension> dims_;
};

}  // namespace oniris
