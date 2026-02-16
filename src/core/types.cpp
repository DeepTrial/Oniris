/**
 * @file types.cpp
 * @brief Core type implementations
 */

#include "core/types.hpp"

#include <sstream>

namespace oniris {

std::string DataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat:        return "float";
        case DataType::kUint8:        return "uint8";
        case DataType::kInt8:         return "int8";
        case DataType::kUint16:       return "uint16";
        case DataType::kInt16:        return "int16";
        case DataType::kInt32:        return "int32";
        case DataType::kInt64:        return "int64";
        case DataType::kString:       return "string";
        case DataType::kBool:         return "bool";
        case DataType::kFloat16:      return "float16";
        case DataType::kDouble:       return "double";
        case DataType::kUint32:       return "uint32";
        case DataType::kUint64:       return "uint64";
        case DataType::kComplex64:    return "complex64";
        case DataType::kComplex128:   return "complex128";
        case DataType::kBFloat16:     return "bfloat16";
        // FP8 types
        case DataType::kFloat8E4M3FN: return "float8e4m3fn";
        case DataType::kFloat8E4M3FNUZ: return "float8e4m3fnuz";
        case DataType::kFloat8E5M2:   return "float8e5m2";
        case DataType::kFloat8E5M2FNUZ: return "float8e5m2fnuz";
        // 4-bit integer types
        case DataType::kUint4:        return "uint4";
        case DataType::kInt4:         return "int4";
        // 4-bit float type
        case DataType::kFloat4E2M1:   return "float4e2m1";
        case DataType::kUndefined:
        default:                      return "undefined";
    }
}

DataType StringToDataType(const std::string& str) {
    // Standard ONNX type names
    if (str == "float" || str == "tensor(float)") return DataType::kFloat;
    if (str == "uint8" || str == "tensor(uint8)") return DataType::kUint8;
    if (str == "int8" || str == "tensor(int8)") return DataType::kInt8;
    if (str == "uint16" || str == "tensor(uint16)") return DataType::kUint16;
    if (str == "int16" || str == "tensor(int16)") return DataType::kInt16;
    if (str == "int32" || str == "tensor(int32)") return DataType::kInt32;
    if (str == "int64" || str == "tensor(int64)") return DataType::kInt64;
    if (str == "string" || str == "tensor(string)") return DataType::kString;
    if (str == "bool" || str == "tensor(bool)") return DataType::kBool;
    if (str == "float16" || str == "tensor(float16)") return DataType::kFloat16;
    if (str == "double" || str == "tensor(double)") return DataType::kDouble;
    if (str == "uint32" || str == "tensor(uint32)") return DataType::kUint32;
    if (str == "uint64" || str == "tensor(uint64)") return DataType::kUint64;
    if (str == "complex64" || str == "tensor(complex64)") return DataType::kComplex64;
    if (str == "complex128" || str == "tensor(complex128)") return DataType::kComplex128;
    if (str == "bfloat16" || str == "tensor(bfloat16)") return DataType::kBFloat16;
    // FP8 types
    if (str == "float8e4m3fn" || str == "tensor(float8e4m3fn)") return DataType::kFloat8E4M3FN;
    if (str == "float8e4m3fnuz" || str == "tensor(float8e4m3fnuz)") return DataType::kFloat8E4M3FNUZ;
    if (str == "float8e5m2" || str == "tensor(float8e5m2)") return DataType::kFloat8E5M2;
    if (str == "float8e5m2fnuz" || str == "tensor(float8e5m2fnuz)") return DataType::kFloat8E5M2FNUZ;
    // 4-bit integer types
    if (str == "uint4" || str == "tensor(uint4)") return DataType::kUint4;
    if (str == "int4" || str == "tensor(int4)") return DataType::kInt4;
    // 4-bit float type
    if (str == "float4e2m1" || str == "tensor(float4e2m1)") return DataType::kFloat4E2M1;
    
    // Alias names for backward compatibility
    if (str == "float32") return DataType::kFloat;
    if (str == "float64") return DataType::kDouble;
    
    return DataType::kUndefined;
}

size_t GetDataTypeSize(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat:        return 4;
        case DataType::kUint8:        return 1;
        case DataType::kInt8:         return 1;
        case DataType::kUint16:       return 2;
        case DataType::kInt16:        return 2;
        case DataType::kInt32:        return 4;
        case DataType::kInt64:        return 8;
        case DataType::kString:       return 0;  // Variable size
        case DataType::kBool:         return 1;
        case DataType::kFloat16:      return 2;
        case DataType::kDouble:       return 8;
        case DataType::kUint32:       return 4;
        case DataType::kUint64:       return 8;
        case DataType::kComplex64:    return 8;
        case DataType::kComplex128:   return 16;
        case DataType::kBFloat16:     return 2;
        // FP8 types - 1 byte each
        case DataType::kFloat8E4M3FN: return 1;
        case DataType::kFloat8E4M3FNUZ: return 1;
        case DataType::kFloat8E5M2:   return 1;
        case DataType::kFloat8E5M2FNUZ: return 1;
        // 4-bit types - packed 2 per byte, so return 0.5 but use 0 to indicate special handling
        case DataType::kUint4:        return 0;  // Special: 4-bit, packed
        case DataType::kInt4:         return 0;  // Special: 4-bit, packed
        // 4-bit float type
        case DataType::kFloat4E2M1:   return 0;  // Special: 4-bit, packed
        case DataType::kUndefined:
        default:                      return 0;
    }
}

// Dimension implementation
std::string Dimension::ToString() const {
    if (is_dynamic_) {
        if (symbol_.empty()) {
            return "?";
        }
        return symbol_;
    }
    return std::to_string(value_);
}

bool Dimension::operator==(const Dimension& other) const {
    if (is_dynamic_ != other.is_dynamic_) {
        return false;
    }
    if (is_dynamic_) {
        return symbol_ == other.symbol_;
    }
    return value_ == other.value_;
}

// Shape implementation
Shape::Shape(std::initializer_list<int64_t> dims) {
    for (int64_t d : dims) {
        dims_.emplace_back(d);
    }
}

bool Shape::IsDynamic() const {
    for (const auto& dim : dims_) {
        if (dim.IsDynamic()) {
            return true;
        }
    }
    return false;
}

std::optional<int64_t> Shape::GetTotalSize() const {
    if (IsDynamic()) {
        return std::nullopt;
    }
    int64_t total = 1;
    for (const auto& dim : dims_) {
        total *= dim.GetStaticValue();
    }
    return total;
}

std::string Shape::ToString() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << dims_[i].ToString();
    }
    oss << "]";
    return oss.str();
}

}  // namespace oniris
