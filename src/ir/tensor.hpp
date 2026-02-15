/**
 * @file tensor.hpp
 * @brief Tensor value and type definition
 */

#pragma once

#include "core/types.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace oniris {

/**
 * @brief Tensor value storage
 */
class Tensor {
public:
    Tensor() = default;
    
    /**
     * @brief Create a tensor with shape and dtype but no data
     * @param shape Tensor shape
     * @param dtype Data type
     */
    Tensor(Shape shape, DataType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {}
    
    /**
     * @brief Create a tensor from raw data
     * @param shape Tensor shape
     * @param dtype Data type
     * @param data Raw data bytes
     */
    Tensor(Shape shape, DataType dtype, std::vector<uint8_t> data)
        : shape_(std::move(shape)), dtype_(dtype), data_(std::move(data)) {}
    
    /// Get tensor shape
    const Shape& GetShape() const { return shape_; }
    Shape& GetShape() { return shape_; }
    void SetShape(const Shape& shape) { shape_ = shape; }
    
    /// Get data type
    DataType GetDataType() const { return dtype_; }
    void SetDataType(DataType dtype) { dtype_ = dtype; }
    
    /// Check if tensor has data
    bool HasData() const { return !data_.empty(); }
    
    /// Get raw data
    const std::vector<uint8_t>& GetData() const { return data_; }
    std::vector<uint8_t>& GetData() { return data_; }
    
    /// Get data as typed pointer
    template<typename T>
    const T* GetDataPtr() const {
        return reinterpret_cast<const T*>(data_.data());
    }
    
    template<typename T>
    T* GetDataPtr() {
        return reinterpret_cast<T*>(data_.data());
    }
    
    /// Get number of elements
    std::optional<int64_t> GetNumElements() const {
        return shape_.GetTotalSize();
    }
    
    /// Calculate required data size in bytes
    std::optional<size_t> GetDataSize() const {
        auto num_elems = GetNumElements();
        if (!num_elems.has_value()) {
            return std::nullopt;
        }
        return *num_elems * GetDataTypeSize(dtype_);
    }
    
    /// Check if tensor is a scalar
    bool IsScalar() const { return shape_.IsScalar(); }
    
    /// Check if tensor has dynamic shape
    bool IsDynamic() const { return shape_.IsDynamic(); }

private:
    Shape shape_;
    DataType dtype_ = DataType::kUnknown;
    std::vector<uint8_t> data_;
};

/**
 * @brief Value information (for inputs/outputs/intermediate values)
 */
struct ValueInfo {
    std::string name;
    Shape shape;
    DataType dtype;
    
    /// Check if shape is inferred
    bool HasInferredShape() const {
        return !shape.GetDims().empty();
    }
};

/**
 * @brief Constant tensor with name
 */
struct ConstantTensor : public ValueInfo {
    Tensor tensor;
};

}  // namespace oniris
