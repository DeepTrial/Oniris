/**
 * @file node.cpp
 * @brief Node implementation
 */

#include "ir/node.hpp"

#include <algorithm>

namespace oniris {

void Node::SetOutputShape(size_t idx, const Shape& shape) {
    if (idx >= output_shapes_.size()) {
        output_shapes_.resize(idx + 1);
    }
    output_shapes_[idx] = shape;
}

bool Node::HasInferredShapes() const {
    return output_shapes_.size() == outputs_.size() &&
           std::all_of(output_shapes_.begin(), output_shapes_.end(),
                      [](const Shape& s) { return !s.GetDims().empty(); });
}

std::shared_ptr<Node> Node::Clone() const {
    auto cloned = std::make_shared<Node>(op_type_, name_);
    cloned->domain_ = domain_;
    cloned->inputs_ = inputs_;
    cloned->outputs_ = outputs_;
    cloned->attributes_ = attributes_;
    cloned->output_shapes_ = output_shapes_;
    return cloned;
}

}  // namespace oniris
