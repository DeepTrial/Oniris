/**
 * @file logger.cpp
 * @brief Logging utilities implementation
 */

#include "core/logger.hpp"

#include <chrono>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace oniris {

namespace {

// Mutex for thread-safe logging
std::mutex g_log_mutex;

// Convert log level to string
const char* LevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug:   return "DEBUG";
        case LogLevel::kInfo:    return "INFO";
        case LogLevel::kWarning: return "WARNING";
        case LogLevel::kError:   return "ERROR";
        case LogLevel::kFatal:   return "FATAL";
        default:                 return "UNKNOWN";
    }
}

// Get current timestamp as string
std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

}  // anonymous namespace

Logger& Logger::GetInstance() {
    static Logger instance;
    return instance;
}

void Logger::Log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level < min_level_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_log_mutex);
    
    // Extract filename from path
    const char* filename = file;
    const char* last_slash = strrchr(file, '/');
    if (last_slash != nullptr) {
        filename = last_slash + 1;
    }
    
    std::cerr << "[" << GetTimestamp() << "] "
              << "[" << LevelToString(level) << "] "
              << "[" << filename << ":" << line << "] "
              << message << std::endl;
    
    if (level == LogLevel::kFatal) {
        std::abort();
    }
}

}  // namespace oniris
