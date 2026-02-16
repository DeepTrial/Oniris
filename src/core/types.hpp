/**
 * @file types.hpp
 * @brief Core type definitions for Oniris
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oniris {

/**
 * @brief Data type enumeration matching ONNX TensorProto types
 * 
 * Values match ONNX TensorProto.DataType enum:
 * https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
 */
enum class DataType {
    kUndefined = 0,
    kFloat = 1,         // float32
    kUint8 = 2,
    kInt8 = 3,
    kUint16 = 4,
    kInt16 = 5,
    kInt32 = 6,
    kInt64 = 7,
    kString = 8,
    kBool = 9,
    kFloat16 = 10,
    kDouble = 11,       // float64
    kUint32 = 12,
    kUint64 = 13,
    kComplex64 = 14,
    kComplex128 = 15,
    kBFloat16 = 16,
    // FP8 types (ONNX 1.14+)
    kFloat8E4M3FN = 17,
    kFloat8E4M3FNUZ = 18,
    kFloat8E5M2 = 19,
    kFloat8E5M2FNUZ = 20,
    // 4-bit integer types (ONNX 1.14+)
    kUint4 = 21,
    kInt4 = 22,
    // 4-bit float type (ONNX 1.16+)
    kFloat4E2M1 = 23,
    // Keep kUnknown for backward compatibility
    kUnknown = 0,
    // Keep kFloat32 alias for backward compatibility
    kFloat32 = 1,
    // Keep kFloat64 alias for backward compatibility  
    kFloat64 = 11,
};

/**
 * @brief Convert DataType to string representation
 * @param dtype The data type
 * @return String representation of the data type
 */
std::string DataTypeToString(DataType dtype);

/**
 * @brief Convert string to DataType
 * @param str The string representation
 * @return The corresponding DataType
 */
DataType StringToDataType(const std::string& str);

/**
 * @brief Get the size in bytes of a data type
 * @param dtype The data type
 * @return Size in bytes, or 0 for variable-size types
 */
size_t GetDataTypeSize(DataType dtype);

/**
 * @brief Shape dimension that can be either static (int64_t) or dynamic (symbolic)
 */
class Dimension {
public:
    explicit Dimension(int64_t value) : symbol_(""), value_(value), is_dynamic_(false) {}
    
    /// Constructor for dynamic dimension with optional symbolic name
    explicit Dimension(const std::string& symbol = "")
        : symbol_(symbol), value_(-1), is_dynamic_(true) {}
    
    /// Check if dimension is dynamic
    bool IsDynamic() const { return is_dynamic_; }
    
    /// Get static value (valid only if !IsDynamic())
    int64_t GetStaticValue() const { return value_; }
    
    /// Get symbolic name (valid only if IsDynamic())
    const std::string& GetSymbolicName() const { return symbol_; }
    
    /// Set static value
    void SetStaticValue(int64_t value) {
        value_ = value;
        is_dynamic_ = false;
        symbol_.clear();
    }
    
    /// Set dynamic with symbol
    void SetDynamic(const std::string& symbol = "") {
        symbol_ = symbol;
        is_dynamic_ = true;
        value_ = -1;
    }
    
    /// Convert to string representation
    std::string ToString() const;
    
    /// Equality comparison
    bool operator==(const Dimension& other) const;
    bool operator!=(const Dimension& other) const { return !(*this == other); }

private:
    std::string symbol_;      ///< Symbolic name for dynamic dimensions
    int64_t value_;           ///< Static value (valid if !is_dynamic_)
    bool is_dynamic_;         ///< Whether dimension is dynamic
};

/**
 * @brief Tensor shape composed of dimensions
 */
class Shape {
public:
    Shape() = default;
    
    /// Constructor from initializer list of static dimensions
    Shape(std::initializer_list<int64_t> dims);
    
    /// Constructor from vector of dimensions
    explicit Shape(std::vector<Dimension> dims) : dims_(std::move(dims)) {}
    
    /// Get number of dimensions
    size_t NumDims() const { return dims_.size(); }
    
    /// Check if shape is empty (scalar)
    bool IsScalar() const { return dims_.empty(); }
    
    /// Check if any dimension is dynamic
    bool IsDynamic() const;
    
    /// Check if shape is fully static
    bool IsStatic() const { return !IsDynamic(); }
    
    /// Get dimension at index
    const Dimension& GetDim(size_t idx) const { return dims_.at(idx); }
    Dimension& GetDim(size_t idx) { return dims_.at(idx); }
    
    /// Get all dimensions
    const std::vector<Dimension>& GetDims() const { return dims_; }
    std::vector<Dimension>& GetDims() { return dims_; }
    
    /// Add a dimension
    void AddDim(const Dimension& dim) { dims_.push_back(dim); }
    void AddDim(int64_t value) { dims_.emplace_back(value); }
    
    /// Get total number of elements (if fully static)
    std::optional<int64_t> GetTotalSize() const;
    
    /// Convert to string representation
    std::string ToString() const;
    
    /// Equality comparison
    bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    std::vector<Dimension> dims_;
};

}  // namespace oniris
