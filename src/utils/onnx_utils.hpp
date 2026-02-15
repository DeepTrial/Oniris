/**
 * @file onnx_utils.hpp
 * @brief ONNX model import/export utilities
 */

#pragma once

#include "ir/model.hpp"

#include <memory>
#include <string>

namespace oniris {
namespace utils {

/**
 * @brief Load an ONNX model from file
 * @param path Path to the ONNX file
 * @return Loaded model, or nullptr on error
 */
std::shared_ptr<Model> LoadModel(const std::string& path);

/**
 * @brief Load an ONNX model from memory buffer
 * @param data Raw bytes of ONNX protobuf
 * @param size Size of data in bytes
 * @return Loaded model, or nullptr on error
 */
std::shared_ptr<Model> LoadModelFromBuffer(const void* data, size_t size);

/**
 * @brief Save a model to ONNX file
 * @param model Model to save
 * @param path Output file path
 * @return true on success
 */
bool SaveModel(const std::shared_ptr<Model>& model, const std::string& path);

/**
 * @brief Serialize a model to bytes
 * @param model Model to serialize
 * @return Serialized bytes, or empty vector on error
 */
std::vector<uint8_t> SerializeModel(const std::shared_ptr<Model>& model);

/**
 * @brief Check if a file is a valid ONNX model
 * @param path Path to check
 * @return true if valid ONNX model
 */
bool IsValidONNXFile(const std::string& path);

/**
 * @brief Get ONNX version information
 * @return Version string
 */
std::string GetONNXVersion();

/**
 * @brief Model info structure
 */
struct ModelInfo {
    std::string producer_name;
    std::string producer_version;
    int64_t ir_version;
    int64_t opset_version;
    size_t num_nodes;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_initializers;
    std::vector<std::string> ops_used;
};

/**
 * @brief Get information about a model
 * @param model Model to analyze
 * @return Model information
 */
ModelInfo GetModelInfo(const std::shared_ptr<Model>& model);

/**
 * @brief Print model summary to stdout
 * @param model Model to print
 */
void PrintModelSummary(const std::shared_ptr<Model>& model);

}  // namespace utils
}  // namespace oniris
