#!/bin/bash
# Lint and format script for Oniris

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

show_help() {
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Commands:"
    echo "  check       Run all linters (default)"
    echo "  format      Format code"
    echo "  cpp         Run C++ linter only"
    echo "  python      Run Python linter only"
    echo "  fix         Auto-fix issues where possible"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  Run all linters"
    echo "  $0 format           Format all code"
    echo "  $0 cpp              Lint C++ code only"
    echo "  $0 python fix       Lint and fix Python code"
}

COMMAND="check"
FIX=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        check|format|cpp|python|fix)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

lint_cpp() {
    echo "Linting C++ code..."
    
    # Check for clang-format
    if command -v clang-format &> /dev/null; then
        if [ "$COMMAND" = "format" ] || [ "$FIX" -eq 1 ]; then
            echo "Formatting C++ files..."
            find "$PROJECT_ROOT/src" -name "*.cpp" -o -name "*.hpp" | \
                xargs clang-format -i 2>/dev/null || echo "Warning: Some files could not be formatted"
        else
            echo "Checking C++ formatting..."
            find "$PROJECT_ROOT/src" -name "*.cpp" -o -name "*.hpp" | \
                xargs clang-format --dry-run --Werror 2>/dev/null || {
                echo "Warning: C++ formatting issues found. Run '$0 format' to fix."
            }
        fi
    else
        echo "Warning: clang-format not found. Skipping C++ formatting."
    fi
    
    # Check for cppcheck
    if command -v cppcheck &> /dev/null; then
        echo "Running cppcheck..."
        cppcheck --enable=all --error-exitcode=0 \
            --suppress=missingInclude \
            "$PROJECT_ROOT/src" 2>/dev/null || true
    fi
}

lint_python() {
    echo "Linting Python code..."
    
    # Black formatter
    if command -v black &> /dev/null; then
        if [ "$COMMAND" = "format" ] || [ "$FIX" -eq 1 ]; then
            echo "Formatting Python files with black..."
            black "$PROJECT_ROOT/python" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/examples" \
                --line-length 100 2>/dev/null || echo "Warning: Some files could not be formatted"
        else
            echo "Checking Python formatting with black..."
            black --check "$PROJECT_ROOT/python" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/examples" \
                --line-length 100 2>/dev/null || {
                echo "Warning: Python formatting issues found. Run '$0 format' to fix."
            }
        fi
    else
        echo "Warning: black not found. Install with: pip install black"
    fi
    
    # Flake8 linter
    if command -v flake8 &> /dev/null; then
        echo "Running flake8..."
        flake8 "$PROJECT_ROOT/python" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/examples" \
            --max-line-length=100 2>/dev/null || true
    else
        echo "Warning: flake8 not found. Install with: pip install flake8"
    fi
    
    # MyPy type checker
    if command -v mypy &> /dev/null; then
        echo "Running mypy..."
        mypy "$PROJECT_ROOT/python/oniris" --ignore-missing-imports 2>/dev/null || true
    else
        echo "Warning: mypy not found. Install with: pip install mypy"
    fi
}

# Main execution
echo "========================================"
echo "Oniris Lint"
echo "========================================"
echo "Command: $COMMAND"
echo ""

case $COMMAND in
    check)
        lint_cpp
        lint_python
        ;;
    format)
        lint_cpp
        lint_python
        echo ""
        echo "Formatting completed!"
        ;;
    cpp)
        lint_cpp
        ;;
    python)
        lint_python
        ;;
    fix)
        FIX=1
        lint_python
        echo ""
        echo "Auto-fix completed!"
        ;;
esac

echo ""
echo "========================================"
echo "Lint completed!"
echo "========================================"
