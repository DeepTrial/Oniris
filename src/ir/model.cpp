/**
 * @file model.cpp
 * @brief Model implementation
 */

#include "ir/model.hpp"

namespace oniris {

std::shared_ptr<Model> Model::Clone() const {
    auto cloned = std::make_shared<Model>(ir_version_);
    cloned->producer_name_ = producer_name_;
    cloned->producer_version_ = producer_version_;
    cloned->domain_ = domain_;
    cloned->model_version_ = model_version_;
    cloned->doc_string_ = doc_string_;
    cloned->opset_imports_ = opset_imports_;
    cloned->metadata_props_ = metadata_props_;
    if (graph_) {
        cloned->graph_ = graph_->Clone();
    }
    return cloned;
}

bool Model::Validate(std::string* error_msg) const {
    if (!graph_) {
        if (error_msg) {
            *error_msg = "Model has no graph";
        }
        return false;
    }
    
    if (opset_imports_.empty()) {
        if (error_msg) {
            *error_msg = "Model has no opset imports";
        }
        return false;
    }
    
    return graph_->Validate(error_msg);
}

}  // namespace oniris
