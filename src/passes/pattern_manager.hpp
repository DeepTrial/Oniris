/**
 * @file pattern_manager.hpp
 * @brief Pattern Manager - Unified pattern management system
 * 
 * Provides centralized management for user-defined patterns including:
 * - Pattern registration and categorization
 * - Pattern validation and metadata
 * - Import/Export from JSON
 * - Pattern versioning
 */

#pragma once

#include "passes/onnx_matcher_style.hpp"
#include "passes/compiler.hpp"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <shared_mutex>

namespace oniris {
namespace passes {

// Forward declarations
class PatternManager;
class PatternRegistry;

/**
 * @brief Pattern category types
 */
enum class PatternCategory {
    kFusion,        // Operator fusion patterns (Conv+ReLU, etc.)
    kOptimization,  // Optimization patterns (constant folding, etc.)
    kQuantization,  // Quantization-related patterns
    kCustom,        // User-defined custom patterns
    kAnalysis,      // Analysis patterns (for profiling, etc.)
    kAll            // All categories
};

/**
 * @brief Convert category to string
 */
std::string PatternCategoryToString(PatternCategory category);

/**
 * @brief Convert string to category
 */
PatternCategory StringToPatternCategory(const std::string& str);

/**
 * @brief Pattern metadata
 */
struct PatternMetadata {
    std::string name;
    std::string description;
    std::string author;
    std::string version{"1.0.0"};
    PatternCategory category{PatternCategory::kCustom};
    std::vector<std::string> tags;
    std::unordered_map<std::string, std::string> attributes;
    std::string created_at;
    std::string modified_at;
    bool enabled{true};
    int priority{0};  // Higher priority = matched first
    
    // Validation
    bool IsValid() const { return !name.empty(); }
};

/**
 * @brief Complete pattern definition with metadata
 */
struct ManagedPattern {
    PatternMetadata metadata;
    PatternDefinition definition;
    
    // Parsed pattern (cached after parsing)
    mutable std::optional<OnnxMatcherPattern> parsed_pattern;
    mutable bool parse_attempted{false};
    
    ManagedPattern() = default;
    ManagedPattern(const std::string& name, const std::string& pattern_str,
                   PatternCategory category = PatternCategory::kCustom);
    
    // Get or parse the pattern
    const std::optional<OnnxMatcherPattern>& GetParsedPattern() const;
    
    // Check if pattern is valid
    bool IsValid() const;
    
    // Get unique ID (name + version)
    std::string GetUniqueId() const {
        return metadata.name + "@" + metadata.version;
    }
};

/**
 * @brief Pattern query/filter criteria
 */
struct PatternQuery {
    std::optional<PatternCategory> category;
    std::vector<std::string> tags;
    std::string name_contains;
    bool enabled_only{true};
    std::optional<int> min_priority;
    std::optional<int> max_priority;
    
    bool Matches(const ManagedPattern& pattern) const;
};

/**
 * @brief Pattern import/export format (JSON)
 */
struct PatternCollection {
    std::string name;
    std::string description;
    std::string version{"1.0.0"};
    std::vector<ManagedPattern> patterns;
    std::unordered_map<std::string, std::string> metadata;
    
    // Serialize to JSON
    std::string ToJson(bool pretty = true) const;
    
    // Load from JSON
    static std::optional<PatternCollection> FromJson(const std::string& json_str);
    
    // Load from file
    static std::optional<PatternCollection> FromFile(const std::string& filepath);
    
    // Save to file
    bool SaveToFile(const std::string& filepath, bool pretty = true) const;
};

/**
 * @brief Pattern statistics
 */
struct PatternStatistics {
    int total_patterns{0};
    int enabled_patterns{0};
    int valid_patterns{0};
    int invalid_patterns{0};
    std::unordered_map<PatternCategory, int> category_counts;
    std::unordered_map<std::string, int> tag_counts;
};

/**
 * @brief Pattern validation result
 */
struct PatternValidationResult {
    bool valid{true};
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    
    void AddError(const std::string& msg) { valid = false; errors.push_back(msg); }
    void AddWarning(const std::string& msg) { warnings.push_back(msg); }
};

/**
 * @brief Pattern Manager - Manages a collection of patterns
 * 
 * Thread-safe (for read operations) pattern management.
 */
class PatternManager {
public:
    PatternManager();
    ~PatternManager() = default;
    
    // Disable copy, enable move
    PatternManager(const PatternManager&) = delete;
    PatternManager& operator=(const PatternManager&) = delete;
    PatternManager(PatternManager&&) = default;
    PatternManager& operator=(PatternManager&&) = default;
    
    // ========================================================================
    // Pattern Registration
    // ========================================================================
    
    /**
     * @brief Register a pattern
     * @param pattern The pattern to register
     * @param overwrite If true, overwrite existing pattern with same name
     * @return true if registration succeeded
     */
    bool RegisterPattern(const ManagedPattern& pattern, bool overwrite = false);
    
    /**
     * @brief Register a simple pattern
     * @param name Pattern name
     * @param pattern_str Pattern string
     * @param category Pattern category
     * @param description Optional description
     * @return true if registration succeeded
     */
    bool RegisterPattern(const std::string& name, const std::string& pattern_str,
                         PatternCategory category = PatternCategory::kCustom,
                         const std::string& description = "");
    
    /**
     * @brief Unregister a pattern
     * @param name Pattern name
     * @return true if pattern was found and removed
     */
    bool UnregisterPattern(const std::string& name);
    
    /**
     * @brief Unregister all patterns
     */
    void ClearPatterns();
    
    /**
     * @brief Unregister patterns by category
     */
    void ClearPatternsByCategory(PatternCategory category);
    
    // ========================================================================
    // Pattern Retrieval
    // ========================================================================
    
    /**
     * @brief Get a pattern by name
     * @param name Pattern name
     * @return Pointer to pattern, or nullptr if not found
     */
    const ManagedPattern* GetPattern(const std::string& name) const;
    
    /**
     * @brief Get all pattern names
     */
    std::vector<std::string> GetPatternNames() const;
    
    /**
     * @brief Get all patterns
     */
    std::vector<const ManagedPattern*> GetAllPatterns() const;
    
    /**
     * @brief Query patterns based on criteria
     */
    std::vector<const ManagedPattern*> QueryPatterns(const PatternQuery& query) const;
    
    /**
     * @brief Get patterns by category
     */
    std::vector<const ManagedPattern*> GetPatternsByCategory(PatternCategory category) const;
    
    /**
     * @brief Get patterns by tag
     */
    std::vector<const ManagedPattern*> GetPatternsByTag(const std::string& tag) const;
    
    // ========================================================================
    // Pattern State Management
    // ========================================================================
    
    /**
     * @brief Enable/disable a pattern
     */
    bool SetPatternEnabled(const std::string& name, bool enabled);
    
    /**
     * @brief Set pattern priority
     */
    bool SetPatternPriority(const std::string& name, int priority);
    
    /**
     * @brief Enable all patterns in a category
     */
    void SetCategoryEnabled(PatternCategory category, bool enabled);
    
    /**
     * @brief Check if pattern exists
     */
    bool HasPattern(const std::string& name) const;
    
    /**
     * @brief Check if pattern is enabled
     */
    bool IsPatternEnabled(const std::string& name) const;
    
    // ========================================================================
    // Pattern Validation
    // ========================================================================
    
    /**
     * @brief Validate a pattern
     */
    static PatternValidationResult ValidatePattern(const ManagedPattern& pattern);
    
    /**
     * @brief Validate all registered patterns
     */
    std::unordered_map<std::string, PatternValidationResult> ValidateAllPatterns() const;
    
    /**
     * @brief Get invalid patterns
     */
    std::vector<std::string> GetInvalidPatterns() const;
    
    // ========================================================================
    // Import/Export
    // ========================================================================
    
    /**
     * @brief Import patterns from a collection
     * @param collection Pattern collection to import
     * @param overwrite If true, overwrite existing patterns
     * @return Number of patterns imported
     */
    int ImportPatterns(const PatternCollection& collection, bool overwrite = false);
    
    /**
     * @brief Import patterns from JSON string
     */
    int ImportPatternsFromJson(const std::string& json_str, bool overwrite = false);
    
    /**
     * @brief Import patterns from file
     */
    int ImportPatternsFromFile(const std::string& filepath, bool overwrite = false);
    
    /**
     * @brief Export all patterns to a collection
     */
    PatternCollection ExportPatterns(const std::string& collection_name = "patterns") const;
    
    /**
     * @brief Export patterns matching query to a collection
     */
    PatternCollection ExportPatterns(const PatternQuery& query, 
                                     const std::string& collection_name = "patterns") const;
    
    /**
     * @brief Export to JSON string
     */
    std::string ExportToJson(bool pretty = true) const;
    
    /**
     * @brief Export to file
     */
    bool ExportToFile(const std::string& filepath, bool pretty = true) const;
    
    // ========================================================================
    // Statistics and Information
    // ========================================================================
    
    /**
     * @brief Get pattern statistics
     */
    PatternStatistics GetStatistics() const;
    
    /**
     * @brief Get number of registered patterns
     */
    size_t GetPatternCount() const;
    
    /**
     * @brief Get number of enabled patterns
     */
    size_t GetEnabledPatternCount() const;
    
    /**
     * @brief Print pattern summary
     */
    void PrintSummary() const;
    
    /**
     * @brief Get unique tags across all patterns
     */
    std::vector<std::string> GetAllTags() const;
    
    // ========================================================================
    // Integration with Compiler
    // ========================================================================
    
    /**
     * @brief Get all enabled patterns as PatternDefinitions for compilation
     */
    std::vector<PatternDefinition> GetEnabledPatternDefinitions() const;
    
    /**
     * @brief Apply patterns to a ModelCompiler
     */
    void ApplyToCompiler(ModelCompiler& compiler) const;
    
    /**
     * @brief Create a ModelCompiler with all enabled patterns
     */
    ModelCompiler CreateCompiler() const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ManagedPattern> patterns_;
    
    // Helper methods
    void UpdateTimestamp(ManagedPattern& pattern);
};

/**
 * @brief Global Pattern Registry - Singleton for application-wide pattern management
 */
class PatternRegistry {
public:
    /**
     * @brief Get the global pattern registry instance
     */
    static PatternRegistry& GetInstance();
    
    /**
     * @brief Get the global pattern manager
     */
    PatternManager& GetManager() { return manager_; }
    const PatternManager& GetManager() const { return manager_; }
    
    // Convenience methods that delegate to manager
    bool RegisterPattern(const ManagedPattern& pattern, bool overwrite = false) {
        return manager_.RegisterPattern(pattern, overwrite);
    }
    
    bool RegisterPattern(const std::string& name, const std::string& pattern_str,
                         PatternCategory category = PatternCategory::kCustom,
                         const std::string& description = "") {
        return manager_.RegisterPattern(name, pattern_str, category, description);
    }
    
    const ManagedPattern* GetPattern(const std::string& name) const {
        return manager_.GetPattern(name);
    }
    
    std::vector<const ManagedPattern*> GetAllPatterns() const {
        return manager_.GetAllPatterns();
    }
    
    void ClearPatterns() { manager_.ClearPatterns(); }
    
    PatternStatistics GetStatistics() const { return manager_.GetStatistics(); }
    
    // Load built-in patterns
    void LoadBuiltinPatterns();
    
    // Load patterns from default locations
    void LoadDefaultPatterns();

public:
    ~PatternRegistry() = default;

private:
    PatternRegistry();
    PatternRegistry(const PatternRegistry&) = delete;
    PatternRegistry& operator=(const PatternRegistry&) = delete;
    
    PatternManager manager_;
};

/**
 * @brief RAII helper for temporary pattern registration
 */
class PatternScope {
public:
    explicit PatternScope(const ManagedPattern& pattern);
    ~PatternScope();
    
    PatternScope(const PatternScope&) = delete;
    PatternScope& operator=(const PatternScope&) = delete;

private:
    std::string pattern_name_;
    bool registered_;
};

/**
 * @brief Convenience function to get global pattern registry
 */
inline PatternRegistry& GetPatternRegistry() {
    return PatternRegistry::GetInstance();
}

/**
 * @brief Predefined pattern collections
 */
namespace PatternCollections {
    // Common fusion patterns
    PatternCollection GetFusionPatterns();
    
    // Optimization patterns
    PatternCollection GetOptimizationPatterns();
    
    // Quantization patterns
    PatternCollection GetQuantizationPatterns();
    
    // All built-in patterns
    PatternCollection GetAllBuiltinPatterns();
}

}  // namespace passes
}  // namespace oniris
