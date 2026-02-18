/**
 * @file simplifier.hpp
 * @brief Model simplification pass (onnxsim-like functionality)
 */

#pragma once

#include "ir/model.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace oniris {
namespace passes {

/**
 * @brief Simplification options
 */
struct SimplifyOptions {
    /// Skip shape inference
    bool skip_shape_inference = false;
    
    /// Skip constant folding
    bool skip_constant_folding = false;
    
    /// Skip converting Constant nodes to initializers
    bool skip_constant_to_initializer = false;
    
    /// Skip dead node elimination
    bool skip_dead_node_elimination = false;
    
    /// Skip identity elimination
    bool skip_identity_elimination = false;
    
    /// Skip shape-related ops elimination
    bool skip_shape_ops_simplification = false;
    
    /// Skip nop transpose elimination
    bool skip_transpose_elimination = false;
    
    /// Skip nop reshape elimination
    bool skip_reshape_elimination = false;
    
    /// Skip nop pad elimination
    bool skip_pad_elimination = false;
    
    /// Skip nop slice elimination
    bool skip_slice_elimination = false;
    
    /// Enable Conv+BN fusion
    bool fuse_conv_bn = true;
    
    /// Enable Conv+ReLU fusion
    bool fuse_conv_relu = true;
    
    /// Enable Gemm+Activation fusion
    bool fuse_gemm_activation = true;
    
    /// Enable Gemm+Bias fusion (fold Add into Gemm's C input)
    bool fuse_gemm_bias = true;
    
    /// Enable QGemm+Activation fusion
    bool fuse_qgemm_activation = true;
    
    /// Fail on unsupported ops (default: false, skip gracefully)
    bool fail_on_unsupported = false;
    
    /// Maximum number of optimization iterations
    int max_iterations = 10;
    
    /// Verbose logging
    bool verbose = false;
};

/**
 * @brief Simplification result
 */
struct SimplifyResult {
    /// Whether simplification succeeded
    bool success = true;
    
    /// Error message if failed
    std::string error_msg;
    
    /// Number of changes made
    int num_changes = 0;
    
    /// Number of iterations performed
    int num_iterations = 0;
    
    /// Names of unsupported ops encountered (if not failing)
    std::vector<std::string> unsupported_ops;
    
    /// Detailed pass statistics
    std::unordered_map<std::string, int> pass_stats;
};

/**
 * @brief Model simplifier (onnxsim-like functionality)
 */
class Simplifier {
public:
    /**
     * @brief Simplify a model
     * @param model The model to simplify
     * @param options Simplification options
     * @return Simplification result
     */
    static SimplifyResult Simplify(const std::shared_ptr<Model>& model,
                                    const SimplifyOptions& options = {});
    
    /**
     * @brief Simplify a graph
     * @param graph The graph to simplify
     * @param options Simplification options
     * @return Simplification result
     */
    static SimplifyResult SimplifyGraph(const std::shared_ptr<Graph>& graph,
                                         const SimplifyOptions& options = {});

    // Basic optimization passes (public for testing)
    static int ConstantFolding(Graph& graph, bool fail_on_unsupported);
    static int EliminateDeadNodes(Graph& graph);
    static int EliminateIdentity(Graph& graph);
    static int EliminateNopTranspose(Graph& graph);
    static int EliminateNopReshape(Graph& graph);
    static int EliminateNopPad(Graph& graph);
    static int EliminateNopSlice(Graph& graph);
    static int EliminateNopResize(Graph& graph);
    static int EliminateSingleInputConcat(Graph& graph);
    static int EliminateUnusedConstants(Graph& graph);
    static int ConvertConstantsToInitializers(Graph& graph);
    static int SimplifyShapeOps(Graph& graph);
    static int MergeConsecutiveOps(Graph& graph);
    
    // Fusion passes (user controllable)
    static int FuseConvBN(Graph& graph);
    static int FuseConvRelu(Graph& graph);
    static int FuseGemmActivations(Graph& graph);
    static int FuseGemmBias(Graph& graph);
    static int FuseQGemmActivations(Graph& graph);
    static int FuseMicrosoftActivations(Graph& graph);
    static int FuseFusedOps(Graph& graph);
    
    // Main execution
    static int RunAllPasses(Graph& graph, const SimplifyOptions& options);

private:
    
    // Helpers
    static bool IsConstant(const Graph& graph, const std::string& name);
    static Tensor GetConstantValue(const Graph& graph, const std::string& name);
    static bool ComputeNode(const Node& node, 
                            const std::vector<Tensor>& inputs,
                            std::vector<Tensor>& outputs,
                            bool fail_on_unsupported);
    static bool CanFoldNode(const Node& node, const Graph& graph);
};

}  // namespace passes
}  // namespace oniris
