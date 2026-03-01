/**
 * @file pattern_manager.cpp
 * @brief Pattern Manager implementation
 */

#include "passes/pattern_manager.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace oniris {
namespace passes {

// ============================================================================
// Helper functions
// ============================================================================

std::string PatternCategoryToString(PatternCategory category) {
    switch (category) {
        case PatternCategory::kFusion: return "fusion";
        case PatternCategory::kOptimization: return "optimization";
        case PatternCategory::kQuantization: return "quantization";
        case PatternCategory::kCustom: return "custom";
        case PatternCategory::kAnalysis: return "analysis";
        case PatternCategory::kAll: return "all";
    }
    return "unknown";
}

PatternCategory StringToPatternCategory(const std::string& str) {
    if (str == "fusion") return PatternCategory::kFusion;
    if (str == "optimization") return PatternCategory::kOptimization;
    if (str == "quantization") return PatternCategory::kQuantization;
    if (str == "custom") return PatternCategory::kCustom;
    if (str == "analysis") return PatternCategory::kAnalysis;
    if (str == "all") return PatternCategory::kAll;
    return PatternCategory::kCustom;
}

static std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// ============================================================================
// ManagedPattern implementation
// ============================================================================

ManagedPattern::ManagedPattern(const std::string& name, const std::string& pattern_str,
                               PatternCategory category)
    : definition(name, pattern_str) {
    metadata.name = name;
    metadata.category = category;
    metadata.created_at = GetCurrentTimestamp();
    metadata.modified_at = metadata.created_at;
}

const std::optional<OnnxMatcherPattern>& ManagedPattern::GetParsedPattern() const {
    if (!parse_attempted) {
        parse_attempted = true;
        auto parsed = OnnxMatcherPattern::FromString(definition.pattern_string);
        if (parsed) {
            parsed_pattern = *parsed;
        }
    }
    return parsed_pattern;
}

bool ManagedPattern::IsValid() const {
    if (!metadata.IsValid()) return false;
    auto parsed = GetParsedPattern();
    return parsed.has_value();
}

// ============================================================================
// PatternQuery implementation
// ============================================================================

bool PatternQuery::Matches(const ManagedPattern& pattern) const {
    // Check category
    if (category.has_value() && pattern.metadata.category != *category) {
        return false;
    }
    
    // Check enabled status
    if (enabled_only && !pattern.metadata.enabled) {
        return false;
    }
    
    // Check name contains
    if (!name_contains.empty()) {
        if (pattern.metadata.name.find(name_contains) == std::string::npos) {
            return false;
        }
    }
    
    // Check tags
    if (!tags.empty()) {
        bool has_tag = false;
        for (const auto& tag : tags) {
            if (std::find(pattern.metadata.tags.begin(), pattern.metadata.tags.end(), tag) 
                != pattern.metadata.tags.end()) {
                has_tag = true;
                break;
            }
        }
        if (!has_tag) return false;
    }
    
    // Check priority range
    if (min_priority.has_value() && pattern.metadata.priority < *min_priority) {
        return false;
    }
    if (max_priority.has_value() && pattern.metadata.priority > *max_priority) {
        return false;
    }
    
    return true;
}

// ============================================================================
// PatternCollection JSON implementation
// ============================================================================

// Forward declarations for JSON helpers
static std::string EscapeJsonString(const std::string& str);
static std::string VectorToJsonArray(const std::vector<std::string>& vec);
static std::string MapToJsonObject(const std::unordered_map<std::string, std::string>& map);

std::string PatternCollection::ToJson(bool pretty) const {
    std::ostringstream oss;
    const std::string indent = pretty ? "  " : "";
    const std::string newline = pretty ? "\n" : "";
    
    auto inc = [&](int level) -> std::string {
        return pretty ? std::string(level * 2, ' ') : "";
    };
    
    oss << "{" << newline;
    oss << inc(1) << "\"name\": \"" << EscapeJsonString(name) << "\"," << newline;
    oss << inc(1) << "\"description\": \"" << EscapeJsonString(description) << "\"," << newline;
    oss << inc(1) << "\"version\": \"" << EscapeJsonString(version) << "\"," << newline;
    oss << inc(1) << "\"metadata\": " << MapToJsonObject(metadata) << "," << newline;
    oss << inc(1) << "\"patterns\": [" << newline;
    
    for (size_t i = 0; i < patterns.size(); ++i) {
        const auto& p = patterns[i];
        if (i > 0) oss << "," << newline;
        
        oss << inc(2) << "{" << newline;
        oss << inc(3) << "\"name\": \"" << EscapeJsonString(p.metadata.name) << "\"," << newline;
        oss << inc(3) << "\"pattern_string\": \"" << EscapeJsonString(p.definition.pattern_string) << "\"," << newline;
        oss << inc(3) << "\"description\": \"" << EscapeJsonString(p.metadata.description) << "\"," << newline;
        oss << inc(3) << "\"category\": \"" << PatternCategoryToString(p.metadata.category) << "\"," << newline;
        oss << inc(3) << "\"version\": \"" << EscapeJsonString(p.metadata.version) << "\"," << newline;
        oss << inc(3) << "\"author\": \"" << EscapeJsonString(p.metadata.author) << "\"," << newline;
        oss << inc(3) << "\"tags\": " << VectorToJsonArray(p.metadata.tags) << "," << newline;
        oss << inc(3) << "\"enabled\": " << (p.metadata.enabled ? "true" : "false") << "," << newline;
        oss << inc(3) << "\"priority\": " << p.metadata.priority << newline;
        oss << inc(2) << "}";
    }
    
    if (!patterns.empty()) oss << newline;
    oss << inc(1) << "]" << newline;
    oss << "}";
    
    return oss.str();
}

std::optional<PatternCollection> PatternCollection::FromJson(const std::string& json_str) {
    PatternCollection collection;
    
    // Helper to extract string value from "key": "value"
    auto extract_string = [&](const std::string& key, size_t start_pos) -> std::pair<std::string, size_t> {
        size_t key_pos = json_str.find("\"" + key + "\"", start_pos);
        if (key_pos == std::string::npos) return {"", start_pos};
        
        size_t colon_pos = json_str.find(":", key_pos);
        if (colon_pos == std::string::npos) return {"", start_pos};
        
        // Skip whitespace and find opening quote
        size_t val_start = colon_pos + 1;
        while (val_start < json_str.size() && (json_str[val_start] == ' ' || json_str[val_start] == '\t' || json_str[val_start] == '\n' || json_str[val_start] == '\r')) {
            val_start++;
        }
        
        if (val_start >= json_str.size() || json_str[val_start] != '"') return {"", start_pos};
        
        val_start++; // Skip opening quote
        size_t val_end = val_start;
        while (val_end < json_str.size() && json_str[val_end] != '"') {
            if (json_str[val_end] == '\\' && val_end + 1 < json_str.size()) {
                val_end += 2; // Skip escaped character
            } else {
                val_end++;
            }
        }
        
        return {json_str.substr(val_start, val_end - val_start), val_end};
    };
    
    // Helper to extract array of strings
    auto extract_string_array = [&](const std::string& key, size_t start_pos) -> std::pair<std::vector<std::string>, size_t> {
        std::vector<std::string> result;
        size_t key_pos = json_str.find("\"" + key + "\"", start_pos);
        if (key_pos == std::string::npos) return {result, start_pos};
        
        size_t bracket_pos = json_str.find("[", key_pos);
        if (bracket_pos == std::string::npos) return {result, start_pos};
        
        size_t pos = bracket_pos + 1;
        while (pos < json_str.size()) {
            // Skip whitespace
            while (pos < json_str.size() && (json_str[pos] == ' ' || json_str[pos] == '\t' || json_str[pos] == '\n' || json_str[pos] == '\r')) pos++;
            
            if (pos >= json_str.size() || json_str[pos] == ']') break;
            
            if (json_str[pos] == '"') {
                pos++;
                size_t end = pos;
                while (end < json_str.size() && json_str[end] != '"') {
                    if (json_str[end] == '\\' && end + 1 < json_str.size()) end += 2;
                    else end++;
                }
                result.push_back(json_str.substr(pos, end - pos));
                pos = end + 1;
            } else {
                pos++;
            }
        }
        
        return {result, pos};
    };
    
    // Helper to extract boolean
    auto extract_bool = [&](const std::string& key, size_t start_pos, bool default_val) -> std::pair<bool, size_t> {
        size_t key_pos = json_str.find("\"" + key + "\"", start_pos);
        if (key_pos == std::string::npos) return {default_val, start_pos};
        
        size_t colon_pos = json_str.find(":", key_pos);
        if (colon_pos == std::string::npos) return {default_val, start_pos};
        
        size_t val_start = colon_pos + 1;
        while (val_start < json_str.size() && (json_str[val_start] == ' ' || json_str[val_start] == '\t' || json_str[val_start] == '\n' || json_str[val_start] == '\r')) val_start++;
        
        if (val_start + 4 <= json_str.size() && json_str.substr(val_start, 4) == "true") return {true, val_start + 4};
        if (val_start + 5 <= json_str.size() && json_str.substr(val_start, 5) == "false") return {false, val_start + 5};
        return {default_val, val_start};
    };
    
    // Helper to extract integer
    auto extract_int = [&](const std::string& key, size_t start_pos, int default_val) -> std::pair<int, size_t> {
        size_t key_pos = json_str.find("\"" + key + "\"", start_pos);
        if (key_pos == std::string::npos) return {default_val, start_pos};
        
        size_t colon_pos = json_str.find(":", key_pos);
        if (colon_pos == std::string::npos) return {default_val, start_pos};
        
        size_t val_start = colon_pos + 1;
        while (val_start < json_str.size() && (json_str[val_start] == ' ' || json_str[val_start] == '\t' || json_str[val_start] == '\n' || json_str[val_start] == '\r')) val_start++;
        
        int sign = 1;
        if (val_start < json_str.size() && json_str[val_start] == '-') {
            sign = -1;
            val_start++;
        }
        
        int value = 0;
        size_t pos = val_start;
        while (pos < json_str.size() && json_str[pos] >= '0' && json_str[pos] <= '9') {
            value = value * 10 + (json_str[pos] - '0');
            pos++;
        }
        
        return {sign * value, pos};
    };
    
    // Extract collection fields
    auto [name_val, name_end] = extract_string("name", 0);
    collection.name = name_val;
    
    auto [desc_val, desc_end] = extract_string("description", 0);
    collection.description = desc_val;
    
    auto [ver_val, ver_end] = extract_string("version", 0);
    collection.version = ver_val;
    
    // Find patterns array
    size_t patterns_pos = json_str.find("\"patterns\"");
    if (patterns_pos == std::string::npos) return collection;
    
    size_t arr_start = json_str.find("[", patterns_pos);
    if (arr_start == std::string::npos) return collection;
    
    // Parse each pattern object
    size_t pos = arr_start + 1;
    int brace_depth = 0;
    bool in_string = false;
    size_t obj_start = 0;
    
    while (pos < json_str.size()) {
        char c = json_str[pos];
        
        if (c == '"' && (pos == 0 || json_str[pos-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (c == '{') {
                if (brace_depth == 0) obj_start = pos;
                brace_depth++;
            } else if (c == '}') {
                brace_depth--;
                if (brace_depth == 0) {
                    // Extract pattern object
                    std::string obj_str = json_str.substr(obj_start, pos - obj_start + 1);
                    
                    ManagedPattern pattern;
                    
                    auto [pname, _] = extract_string("name", obj_start);
                    pattern.metadata.name = pname;
                    pattern.definition.name = pname;
                    
                    auto [pstr, __] = extract_string("pattern_string", obj_start);
                    // Replace \n with actual newlines
                    std::string actual_pattern = pstr;
                    size_t pos = 0;
                    while ((pos = actual_pattern.find("\\n", pos)) != std::string::npos) {
                        actual_pattern.replace(pos, 2, "\n");
                        pos += 1;
                    }
                    pattern.definition.pattern_string = actual_pattern;
                    
                    auto [pdesc, ___] = extract_string("description", obj_start);
                    pattern.metadata.description = pdesc;
                    
                    auto [pcat, ____] = extract_string("category", obj_start);
                    pattern.metadata.category = StringToPatternCategory(pcat);
                    
                    auto [pver, _____] = extract_string("version", obj_start);
                    pattern.metadata.version = pver;
                    
                    auto [pauth, ______] = extract_string("author", obj_start);
                    pattern.metadata.author = pauth;
                    
                    auto [ptags, _______] = extract_string_array("tags", obj_start);
                    pattern.metadata.tags = ptags;
                    
                    auto [penabled, ________] = extract_bool("enabled", obj_start, true);
                    pattern.metadata.enabled = penabled;
                    
                    auto [ppriority, _________] = extract_int("priority", obj_start, 0);
                    pattern.metadata.priority = ppriority;
                    
                    collection.patterns.push_back(std::move(pattern));
                }
            } else if (c == ']' && brace_depth == 0) {
                break;
            }
        }
        pos++;
    }
    
    return collection;
}

std::optional<PatternCollection> PatternCollection::FromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return std::nullopt;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return FromJson(buffer.str());
}

bool PatternCollection::SaveToFile(const std::string& filepath, bool pretty) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    file << ToJson(pretty);
    return file.good();
}

// ============================================================================
// PatternManager implementation
// ============================================================================

PatternManager::PatternManager() = default;

bool PatternManager::RegisterPattern(const ManagedPattern& pattern, bool overwrite) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(pattern.metadata.name);
    if (it != patterns_.end() && !overwrite) {
        ONIRIS_WARNING << "Pattern '" << pattern.metadata.name << "' already exists. Use overwrite=true to replace.";
        return false;
    }
    
    ManagedPattern copy = pattern;
    UpdateTimestamp(copy);
    patterns_[pattern.metadata.name] = std::move(copy);
    
    ONIRIS_INFO << "Registered pattern: " << pattern.metadata.name 
                << " (category: " << PatternCategoryToString(pattern.metadata.category) << ")";
    return true;
}

bool PatternManager::RegisterPattern(const std::string& name, const std::string& pattern_str,
                                     PatternCategory category, const std::string& description) {
    ManagedPattern pattern(name, pattern_str, category);
    pattern.metadata.description = description;
    return RegisterPattern(pattern);
}

bool PatternManager::UnregisterPattern(const std::string& name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(name);
    if (it == patterns_.end()) {
        return false;
    }
    
    patterns_.erase(it);
    ONIRIS_INFO << "Unregistered pattern: " << name;
    return true;
}

void PatternManager::ClearPatterns() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    patterns_.clear();
    ONIRIS_INFO << "Cleared all patterns";
}

void PatternManager::ClearPatternsByCategory(PatternCategory category) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    for (auto it = patterns_.begin(); it != patterns_.end();) {
        if (it->second.metadata.category == category) {
            it = patterns_.erase(it);
        } else {
            ++it;
        }
    }
    
    ONIRIS_INFO << "Cleared patterns in category: " << PatternCategoryToString(category);
}

const ManagedPattern* PatternManager::GetPattern(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(name);
    if (it != patterns_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> PatternManager::GetPatternNames() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<std::string> names;
    names.reserve(patterns_.size());
    for (const auto& [name, _] : patterns_) {
        names.push_back(name);
    }
    return names;
}

std::vector<const ManagedPattern*> PatternManager::GetAllPatterns() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<const ManagedPattern*> result;
    result.reserve(patterns_.size());
    for (const auto& [_, pattern] : patterns_) {
        result.push_back(&pattern);
    }
    
    // Sort by priority (higher first)
    std::sort(result.begin(), result.end(), 
              [](const ManagedPattern* a, const ManagedPattern* b) {
                  return a->metadata.priority > b->metadata.priority;
              });
    
    return result;
}

std::vector<const ManagedPattern*> PatternManager::QueryPatterns(const PatternQuery& query) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<const ManagedPattern*> result;
    for (const auto& [_, pattern] : patterns_) {
        if (query.Matches(pattern)) {
            result.push_back(&pattern);
        }
    }
    
    // Sort by priority (higher first)
    std::sort(result.begin(), result.end(), 
              [](const ManagedPattern* a, const ManagedPattern* b) {
                  return a->metadata.priority > b->metadata.priority;
              });
    
    return result;
}

std::vector<const ManagedPattern*> PatternManager::GetPatternsByCategory(PatternCategory category) const {
    PatternQuery query;
    query.category = category;
    return QueryPatterns(query);
}

std::vector<const ManagedPattern*> PatternManager::GetPatternsByTag(const std::string& tag) const {
    PatternQuery query;
    query.tags = {tag};
    return QueryPatterns(query);
}

bool PatternManager::SetPatternEnabled(const std::string& name, bool enabled) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(name);
    if (it == patterns_.end()) {
        return false;
    }
    
    it->second.metadata.enabled = enabled;
    it->second.metadata.modified_at = GetCurrentTimestamp();
    return true;
}

bool PatternManager::SetPatternPriority(const std::string& name, int priority) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(name);
    if (it == patterns_.end()) {
        return false;
    }
    
    it->second.metadata.priority = priority;
    it->second.metadata.modified_at = GetCurrentTimestamp();
    return true;
}

void PatternManager::SetCategoryEnabled(PatternCategory category, bool enabled) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    for (auto& [_, pattern] : patterns_) {
        if (pattern.metadata.category == category) {
            pattern.metadata.enabled = enabled;
            pattern.metadata.modified_at = GetCurrentTimestamp();
        }
    }
}

bool PatternManager::HasPattern(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return patterns_.find(name) != patterns_.end();
}

bool PatternManager::IsPatternEnabled(const std::string& name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    auto it = patterns_.find(name);
    if (it != patterns_.end()) {
        return it->second.metadata.enabled;
    }
    return false;
}

PatternValidationResult PatternManager::ValidatePattern(const ManagedPattern& pattern) {
    PatternValidationResult result;
    
    if (pattern.metadata.name.empty()) {
        result.AddError("Pattern name is empty");
    }
    
    if (pattern.definition.pattern_string.empty()) {
        result.AddError("Pattern string is empty");
    } else {
        std::string error_msg;
        PatternDefinition def = pattern.definition;
        if (!def.Parse(&error_msg)) {
            result.AddError("Failed to parse pattern: " + error_msg);
        }
    }
    
    return result;
}

std::unordered_map<std::string, PatternValidationResult> PatternManager::ValidateAllPatterns() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::unordered_map<std::string, PatternValidationResult> results;
    for (const auto& [name, pattern] : patterns_) {
        results[name] = ValidatePattern(pattern);
    }
    return results;
}

std::vector<std::string> PatternManager::GetInvalidPatterns() const {
    auto validations = ValidateAllPatterns();
    
    std::vector<std::string> invalid;
    for (const auto& [name, result] : validations) {
        if (!result.valid) {
            invalid.push_back(name);
        }
    }
    return invalid;
}

int PatternManager::ImportPatterns(const PatternCollection& collection, bool overwrite) {
    int count = 0;
    for (const auto& pattern : collection.patterns) {
        if (RegisterPattern(pattern, overwrite)) {
            ++count;
        }
    }
    ONIRIS_INFO << "Imported " << count << " patterns from collection: " << collection.name;
    return count;
}

int PatternManager::ImportPatternsFromJson(const std::string& json_str, bool overwrite) {
    auto collection = PatternCollection::FromJson(json_str);
    if (!collection) {
        ONIRIS_ERROR << "Failed to parse pattern JSON";
        return 0;
    }
    return ImportPatterns(*collection, overwrite);
}

int PatternManager::ImportPatternsFromFile(const std::string& filepath, bool overwrite) {
    auto collection = PatternCollection::FromFile(filepath);
    if (!collection) {
        ONIRIS_ERROR << "Failed to load pattern file: " << filepath;
        return 0;
    }
    return ImportPatterns(*collection, overwrite);
}

PatternCollection PatternManager::ExportPatterns(const std::string& collection_name) const {
    PatternQuery query;
    query.enabled_only = false;  // Export all patterns
    return ExportPatterns(query, collection_name);
}

PatternCollection PatternManager::ExportPatterns(const PatternQuery& query, 
                                                  const std::string& collection_name) const {
    PatternCollection collection;
    collection.name = collection_name;
    collection.description = "Exported pattern collection";
    
    auto patterns = QueryPatterns(query);
    for (const auto* p : patterns) {
        collection.patterns.push_back(*p);
    }
    
    return collection;
}

std::string PatternManager::ExportToJson(bool pretty) const {
    return ExportPatterns().ToJson(pretty);
}

bool PatternManager::ExportToFile(const std::string& filepath, bool pretty) const {
    return ExportPatterns().SaveToFile(filepath, pretty);
}

PatternStatistics PatternManager::GetStatistics() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    PatternStatistics stats;
    stats.total_patterns = static_cast<int>(patterns_.size());
    
    for (const auto& [_, pattern] : patterns_) {
        if (pattern.metadata.enabled) {
            stats.enabled_patterns++;
        }
        if (pattern.IsValid()) {
            stats.valid_patterns++;
        } else {
            stats.invalid_patterns++;
        }
        
        stats.category_counts[pattern.metadata.category]++;
        
        for (const auto& tag : pattern.metadata.tags) {
            stats.tag_counts[tag]++;
        }
    }
    
    return stats;
}

size_t PatternManager::GetPatternCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return patterns_.size();
}

size_t PatternManager::GetEnabledPatternCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    size_t count = 0;
    for (const auto& [_, pattern] : patterns_) {
        if (pattern.metadata.enabled) {
            ++count;
        }
    }
    return count;
}

void PatternManager::PrintSummary() const {
    auto stats = GetStatistics();
    
    std::cout << "\n=== Pattern Manager Summary ===\n";
    std::cout << "Total patterns: " << stats.total_patterns << "\n";
    std::cout << "Enabled patterns: " << stats.enabled_patterns << "\n";
    std::cout << "Valid patterns: " << stats.valid_patterns << "\n";
    std::cout << "Invalid patterns: " << stats.invalid_patterns << "\n";
    
    std::cout << "\nBy Category:\n";
    for (const auto& [cat, count] : stats.category_counts) {
        std::cout << "  " << PatternCategoryToString(cat) << ": " << count << "\n";
    }
    
    std::cout << "\nBy Tag:\n";
    for (const auto& [tag, count] : stats.tag_counts) {
        std::cout << "  " << tag << ": " << count << "\n";
    }
    std::cout << "================================\n\n";
}

std::vector<std::string> PatternManager::GetAllTags() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::unordered_set<std::string> tags;
    for (const auto& [_, pattern] : patterns_) {
        for (const auto& tag : pattern.metadata.tags) {
            tags.insert(tag);
        }
    }
    
    return std::vector<std::string>(tags.begin(), tags.end());
}

std::vector<PatternDefinition> PatternManager::GetEnabledPatternDefinitions() const {
    auto patterns = GetAllPatterns();
    
    std::vector<PatternDefinition> definitions;
    for (const auto* pattern : patterns) {
        if (pattern->metadata.enabled && pattern->IsValid()) {
            definitions.push_back(pattern->definition);
        }
    }
    
    return definitions;
}

void PatternManager::ApplyToCompiler(ModelCompiler& compiler) const {
    auto definitions = GetEnabledPatternDefinitions();
    compiler.AddPatterns(definitions);
    ONIRIS_INFO << "Applied " << definitions.size() << " patterns to compiler";
}

ModelCompiler PatternManager::CreateCompiler() const {
    ModelCompiler compiler;
    ApplyToCompiler(compiler);
    return compiler;
}

void PatternManager::UpdateTimestamp(ManagedPattern& pattern) {
    pattern.metadata.modified_at = GetCurrentTimestamp();
}

// ============================================================================
// JSON Helpers
// ============================================================================

static std::string EscapeJsonString(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c >= 0x20 && c <= 0x7E) {
                    oss << c;
                } else {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)(unsigned char)c;
                }
        }
    }
    return oss.str();
}

static std::string VectorToJsonArray(const std::vector<std::string>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "\"" << EscapeJsonString(vec[i]) << "\"";
    }
    oss << "]";
    return oss.str();
}

static std::string MapToJsonObject(const std::unordered_map<std::string, std::string>& map) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& [key, value] : map) {
        if (!first) oss << ", ";
        first = false;
        oss << "\"" << EscapeJsonString(key) << "\": \"" << EscapeJsonString(value) << "\"";
    }
    oss << "}";
    return oss.str();
}

// ============================================================================
// PatternRegistry implementation
// ============================================================================

PatternRegistry& PatternRegistry::GetInstance() {
    static PatternRegistry instance;
    return instance;
}

PatternRegistry::PatternRegistry() {
    // Load built-in patterns on first access
    LoadBuiltinPatterns();
}

void PatternRegistry::LoadBuiltinPatterns() {
    // Load common patterns
    auto patterns = GetCommonPatterns();
    for (const auto& p : patterns) {
        ManagedPattern mp(p.name, p.pattern_string, PatternCategory::kFusion);
        mp.metadata.description = "Built-in fusion pattern";
        RegisterPattern(mp, false);  // Don't overwrite if already exists
    }
}

void PatternRegistry::LoadDefaultPatterns() {
    // This could load from default file locations
    // For now, just ensure built-in patterns are loaded
    LoadBuiltinPatterns();
}

// ============================================================================
// PatternScope implementation
// ============================================================================

PatternScope::PatternScope(const ManagedPattern& pattern) 
    : pattern_name_(pattern.metadata.name) {
    registered_ = GetPatternRegistry().RegisterPattern(pattern);
}

PatternScope::~PatternScope() {
    if (registered_) {
        GetPatternRegistry().GetManager().UnregisterPattern(pattern_name_);
    }
}

// ============================================================================
// PatternCollections implementation
// ============================================================================

namespace PatternCollections {

PatternCollection GetFusionPatterns() {
    PatternCollection collection;
    collection.name = "fusion";
    collection.description = "Common operator fusion patterns";
    
    collection.patterns.emplace_back("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)", PatternCategory::kFusion);
    collection.patterns.emplace_back("ConvBnRelu", "Conv(?, c0)\nBatchNormalization(c0, bn0)\nRelu(bn0, ?)", PatternCategory::kFusion);
    collection.patterns.emplace_back("ConvBn", "Conv(?, c0)\nBatchNormalization(c0, ?)", PatternCategory::kFusion);
    collection.patterns.emplace_back("GemmRelu", "Gemm(?, g0)\nRelu(g0, ?)", PatternCategory::kFusion);
    
    return collection;
}

PatternCollection GetOptimizationPatterns() {
    PatternCollection collection;
    collection.name = "optimization";
    collection.description = "Optimization opportunity patterns";
    
    collection.patterns.emplace_back("Identity", "?(?, ?)\nIdentity(?, ?)", PatternCategory::kOptimization);
    collection.patterns.emplace_back("ReshapeReshape", "Reshape(?, r0)\nReshape(r0, ?)", PatternCategory::kOptimization);
    
    return collection;
}

PatternCollection GetQuantizationPatterns() {
    PatternCollection collection;
    collection.name = "quantization";
    collection.description = "Quantization-related patterns";
    
    collection.patterns.emplace_back("QConv", "DequantizeLinear(?, d0)\nConv(d0, ?)\nQuantizeLinear(?, ?)", PatternCategory::kQuantization);
    
    return collection;
}

PatternCollection GetAllBuiltinPatterns() {
    PatternCollection collection;
    collection.name = "builtin";
    collection.description = "All built-in patterns";
    
    auto fusion = GetFusionPatterns();
    auto opt = GetOptimizationPatterns();
    auto quant = GetQuantizationPatterns();
    
    collection.patterns.insert(collection.patterns.end(), 
                               fusion.patterns.begin(), fusion.patterns.end());
    collection.patterns.insert(collection.patterns.end(), 
                               opt.patterns.begin(), opt.patterns.end());
    collection.patterns.insert(collection.patterns.end(), 
                               quant.patterns.begin(), quant.patterns.end());
    
    return collection;
}

}  // namespace PatternCollections

}  // namespace passes
}  // namespace oniris
