/**
 * @file model.hpp
 * @brief ONNX Model representation
 */

#pragma once

#include "ir/graph.hpp"

#include <memory>
#include <string>
#include <vector>

namespace oniris {

/**
 * @brief ONNX operator set information
 */
struct OpsetImport {
    std::string domain;
    int64_t version;
};

/**
 * @brief ONNX Model representation
 */
class Model : public std::enable_shared_from_this<Model> {
public:
    Model() = default;
    
    /**
     * @brief Create a model with IR version
     * @param ir_version ONNX IR version
     */
    explicit Model(int64_t ir_version) : ir_version_(ir_version) {}
    
    /// Get IR version
    int64_t GetIRVersion() const { return ir_version_; }
    void SetIRVersion(int64_t version) { ir_version_ = version; }
    
    /// Get producer name
    const std::string& GetProducerName() const { return producer_name_; }
    void SetProducerName(const std::string& name) { producer_name_ = name; }
    
    /// Get producer version
    const std::string& GetProducerVersion() const { return producer_version_; }
    void SetProducerVersion(const std::string& version) { producer_version_ = version; }
    
    /// Get domain
    const std::string& GetDomain() const { return domain_; }
    void SetDomain(const std::string& domain) { domain_ = domain; }
    
    /// Get model version
    int64_t GetModelVersion() const { return model_version_; }
    void SetModelVersion(int64_t version) { model_version_ = version; }
    
    /// Get doc string
    const std::string& GetDocString() const { return doc_string_; }
    void SetDocString(const std::string& doc) { doc_string_ = doc; }
    
    /// Get opset imports
    const std::vector<OpsetImport>& GetOpsetImports() const { return opset_imports_; }
    std::vector<OpsetImport>& GetOpsetImports() { return opset_imports_; }
    void AddOpsetImport(const OpsetImport& opset) { opset_imports_.push_back(opset); }
    void SetOpsetImports(const std::vector<OpsetImport>& opsets) { opset_imports_ = opsets; }
    
    /// Get main graph
    std::shared_ptr<Graph> GetGraph() const { return graph_; }
    void SetGraph(std::shared_ptr<Graph> graph) { graph_ = std::move(graph); }
    
    /// Create new main graph
    std::shared_ptr<Graph> CreateGraph(const std::string& name = "") {
        graph_ = std::make_shared<Graph>(name);
        return graph_;
    }
    
    /// Get metadata props
    const std::unordered_map<std::string, std::string>& GetMetadataProps() const {
        return metadata_props_;
    }
    
    void SetMetadataProp(const std::string& key, const std::string& value) {
        metadata_props_[key] = value;
    }
    
    /// Clone the model
    std::shared_ptr<Model> Clone() const;
    
    /// Validate model structure
    bool Validate(std::string* error_msg = nullptr) const;

private:
    int64_t ir_version_ = 8;  // Default to ONNX IR version 8
    std::string producer_name_ = "oniris";
    std::string producer_version_ = "0.1.0";
    std::string domain_;
    int64_t model_version_ = 0;
    std::string doc_string_;
    std::vector<OpsetImport> opset_imports_;
    std::shared_ptr<Graph> graph_;
    std::unordered_map<std::string, std::string> metadata_props_;
};

}  // namespace oniris
