/**
 * @file logger.hpp
 * @brief Logging utilities for Oniris
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>

namespace oniris {

/**
 * @brief Log level enumeration
 */
enum class LogLevel {
    kDebug = 0,
    kInfo = 1,
    kWarning = 2,
    kError = 3,
    kFatal = 4,
};

/**
 * @brief Simple logging utility
 */
class Logger {
public:
    /**
     * @brief Get the global logger instance
     * @return Reference to the global logger
     */
    static Logger& GetInstance();
    
    /**
     * @brief Set the global log level
     * @param level Minimum level to log
     */
    void SetLevel(LogLevel level) { min_level_ = level; }
    
    /**
     * @brief Get the current log level
     * @return Current minimum log level
     */
    LogLevel GetLevel() const { return min_level_; }
    
    /**
     * @brief Log a message
     * @param level Log level
     * @param message Message to log
     * @param file Source file
     * @param line Line number
     */
    void Log(LogLevel level, const std::string& message, const char* file, int line);

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    LogLevel min_level_ = LogLevel::kInfo;
};

/**
 * @brief Log stream helper class
 */
class LogStream {
public:
    LogStream(LogLevel level, const char* file, int line)
        : level_(level), file_(file), line_(line) {}
    
    ~LogStream() {
        Logger::GetInstance().Log(level_, stream_.str(), file_, line_);
    }
    
    template<typename T>
    LogStream& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }

private:
    LogLevel level_;
    const char* file_;
    int line_;
    std::ostringstream stream_;
};

// Logging macros
#define ONIRIS_LOG(level) \
    if (level < ::oniris::Logger::GetInstance().GetLevel()) ; \
    else ::oniris::LogStream(level, __FILE__, __LINE__)

#define ONIRIS_DEBUG ONIRIS_LOG(::oniris::LogLevel::kDebug)
#define ONIRIS_INFO ONIRIS_LOG(::oniris::LogLevel::kInfo)
#define ONIRIS_WARNING ONIRIS_LOG(::oniris::LogLevel::kWarning)
#define ONIRIS_ERROR ONIRIS_LOG(::oniris::LogLevel::kError)
#define ONIRIS_FATAL ONIRIS_LOG(::oniris::LogLevel::kFatal)

}  // namespace oniris
