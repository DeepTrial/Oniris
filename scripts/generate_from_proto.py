#!/usr/bin/env python3
"""
Generate C++ IR definitions directly from ONNX proto files.

Usage:
    generate_from_proto.py [path/to/onnx.proto] [output_dir]

If onnx.proto path is not provided, will search in common locations:
    - CMake downloaded: build/third_party/onnx/onnx.proto
    - System install: /usr/share/onnx/onnx.proto
    - Local third_party: third_party/onnx/onnx.proto
"""

import sys
import re
import os
from pathlib import Path


def find_proto_file(proto_path=None):
    """Find onnx.proto file from various sources"""
    if proto_path and Path(proto_path).exists():
        return Path(proto_path)
    
    # Search paths in order of preference
    search_paths = [
        # CMake downloaded (current build dir)
        Path.cwd() / "build" / "third_party" / "onnx" / "onnx.proto",
        Path.cwd() / "third_party" / "onnx" / "onnx.proto",
        # System install locations
        Path("/usr/share/onnx/onnx.proto"),
        Path("/usr/local/share/onnx/onnx.proto"),
        # Python package location
        Path(__file__).parent.parent / "build" / "third_party" / "onnx" / "onnx.proto",
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"Found onnx.proto: {path}")
            return path
    
    return None


def parse_proto_enum(content, enum_name):
    """Parse enum definition from proto file content"""
    pattern = rf'enum\s+{enum_name}\s+{{([^}}]+)}}'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    
    enum_content = match.group(1)
    values = []
    
    for line in enum_content.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        
        match = re.match(r'(\w+)\s*=\s*(\d+)\s*;', line)
        if match:
            name = match.group(1)
            value = int(match.group(2))
            values.append((name, value))
    
    return values


def generate_dtype_from_proto(proto_path, output_dir):
    """Generate DataType from onnx.proto file"""
    
    # Find proto file
    proto_file = find_proto_file(proto_path)
    if not proto_file:
        print("Error: Could not find onnx.proto file")
        if proto_path:
            print(f"  Specified path not found: {proto_path}")
        print("\nPlease provide path to onnx.proto:")
        print("  python generate_from_proto.py /path/to/onnx.proto")
        sys.exit(1)
    
    content = proto_file.read_text()
    
    # Parse DataType enum
    dtypes = parse_proto_enum(content, 'DataType')
    
    if not dtypes:
        print(f"Error: Could not find DataType enum in {proto_file}")
        sys.exit(1)
    
    print(f"Found {len(dtypes)} data types in {proto_file.name}")
    
    # Extract version info from proto
    version_match = re.search(r'IR_VERSION\s*=\s*(0x[0-9A-Fa-f]+);', content)
    ir_version = version_match.group(1) if version_match else "unknown"
    
    # Mapping from ONNX name to C++ name
    onnx_to_cpp = {
        'UNDEFINED': 'kUndefined',
        'FLOAT': 'kFloat',
        'UINT8': 'kUint8',
        'INT8': 'kInt8',
        'UINT16': 'kUint16',
        'INT16': 'kInt16',
        'INT32': 'kInt32',
        'INT64': 'kInt64',
        'STRING': 'kString',
        'BOOL': 'kBool',
        'FLOAT16': 'kFloat16',
        'DOUBLE': 'kDouble',
        'UINT32': 'kUint32',
        'UINT64': 'kUint64',
        'COMPLEX64': 'kComplex64',
        'COMPLEX128': 'kComplex128',
        'BFLOAT16': 'kBFloat16',
        'FLOAT8E4M3FN': 'kFloat8E4M3FN',
        'FLOAT8E4M3FNUZ': 'kFloat8E4M3FNUZ',
        'FLOAT8E5M2': 'kFloat8E5M2',
        'FLOAT8E5M2FNUZ': 'kFloat8E5M2FNUZ',
        'FLOAT8E8M0': 'kFloat8E8M0',
        'UINT4': 'kUint4',
        'INT4': 'kInt4',
        'UINT2': 'kUint2',
        'INT2': 'kInt2',
        'FLOAT4E2M1': 'kFloat4E2M1',
    }
    
    # Build value -> cpp name mapping
    value_to_cpp = {}
    for onnx_name, value in dtypes:
        cpp_name = onnx_to_cpp.get(onnx_name, 'k' + onnx_name.title())
        value_to_cpp[value] = cpp_name
    
    # Generate types.hpp
    hpp_lines = [
        '/**',
        ' * @file types.hpp',
        ' * @brief Core type definitions for Oniris',
        ' * ',
        f' * AUTO-GENERATED from ONNX proto file (IR_VERSION: {ir_version})',
        ' * Do not edit manually. Run scripts/generate_from_proto.py to regenerate.',
        ' */',
        '',
        '#pragma once',
        '',
        '#include <cstdint>',
        '#include <optional>',
        '#include <string>',
        '#include <vector>',
        '',
        'namespace oniris {',
        '',
        '/**',
        ' * @brief Data type enumeration matching ONNX TensorProto.DataType',
        f' * IR_VERSION: {ir_version}',
        ' */',
        'enum class DataType : int {',
        '    // ONNX standard types (auto-generated from onnx.proto)',
    ]
    
    for value, cpp_name in sorted(value_to_cpp.items()):
        hpp_lines.append(f'    {cpp_name} = {value},')
    
    hpp_lines.extend([
        '    ',
        '    // Aliases for backward compatibility',
        '    kUnknown = kUndefined,',
        '    kFloat32 = kFloat,',
        '    kFloat64 = kDouble,',
        '};',
        '',
        'std::string DataTypeToString(DataType dtype);',
        'DataType StringToDataType(const std::string& str);',
        'size_t GetDataTypeSize(DataType dtype);',
        '',
        'class Dimension {',
        'public:',
        '    explicit Dimension(int64_t value) : symbol_(""), value_(value), is_dynamic_(false) {}',
        '    explicit Dimension(const std::string& symbol = "")',
        '        : symbol_(symbol), value_(-1), is_dynamic_(true) {}',
        '    bool IsDynamic() const { return is_dynamic_; }',
        '    int64_t GetStaticValue() const { return value_; }',
        '    const std::string& GetSymbolicName() const { return symbol_; }',
        '    void SetStaticValue(int64_t value) {',
        '        value_ = value;',
        '        is_dynamic_ = false;',
        '        symbol_.clear();',
        '    }',
        '    void SetDynamic(const std::string& symbol = "") {',
        '        symbol_ = symbol;',
        '        is_dynamic_ = true;',
        '        value_ = -1;',
        '    }',
        '    std::string ToString() const;',
        '    bool operator==(const Dimension& other) const;',
        '    bool operator!=(const Dimension& other) const { return !(*this == other); }',
        '',
        'private:',
        '    std::string symbol_;',
        '    int64_t value_;',
        '    bool is_dynamic_;',
        '};',
        '',
        'class Shape {',
        'public:',
        '    Shape() = default;',
        '    Shape(std::initializer_list<int64_t> dims);',
        '    explicit Shape(std::vector<Dimension> dims) : dims_(std::move(dims)) {}',
        '    size_t NumDims() const { return dims_.size(); }',
        '    bool IsScalar() const { return dims_.empty(); }',
        '    bool IsDynamic() const;',
        '    bool IsStatic() const { return !IsDynamic(); }',
        '    const Dimension& GetDim(size_t idx) const { return dims_.at(idx); }',
        '    Dimension& GetDim(size_t idx) { return dims_.at(idx); }',
        '    const std::vector<Dimension>& GetDims() const { return dims_; }',
        '    std::vector<Dimension>& GetDims() { return dims_; }',
        '    void AddDim(const Dimension& dim) { dims_.push_back(dim); }',
        '    void AddDim(int64_t value) { dims_.emplace_back(value); }',
        '    std::optional<int64_t> GetTotalSize() const;',
        '    std::string ToString() const;',
        '    bool operator==(const Shape& other) const { return dims_ == other.dims_; }',
        '    bool operator!=(const Shape& other) const { return !(*this == other); }',
        '',
        'private:',
        '    std::vector<Dimension> dims_;',
        '};',
        '',
        '}  // namespace oniris',
        '',
    ])
    
    # Generate types.cpp
    cpp_lines = [
        '/**',
        ' * @file types.cpp',
        ' * @brief Core type implementations',
        ' * ',
        f' * AUTO-GENERATED from ONNX proto file (IR_VERSION: {ir_version})',
        ' * Do not edit manually. Run scripts/generate_from_proto.py to regenerate.',
        ' */',
        '',
        '#include "core/types.hpp"',
        '#include <sstream>',
        '',
        'namespace oniris {',
        '',
        'std::string DataTypeToString(DataType dtype) {',
        '    switch (dtype) {',
    ]
    
    name_to_onnx = {v: k for k, v in onnx_to_cpp.items()}
    for value, cpp_name in sorted(value_to_cpp.items()):
        onnx_name = name_to_onnx.get(cpp_name, cpp_name[1:].upper())
        lower_name = onnx_name.lower().replace("_", "")
        cpp_lines.append(f'        case DataType::{cpp_name}: return "{lower_name}";')
    
    cpp_lines.extend([
        '        default: return "undefined";',
        '    }',
        '}',
        '',
        'DataType StringToDataType(const std::string& str) {',
    ])
    
    for value, cpp_name in sorted(value_to_cpp.items(), key=lambda x: x[1]):
        onnx_name = name_to_onnx.get(cpp_name, cpp_name[1:].upper())
        lower_name = onnx_name.lower().replace("_", "")
        cpp_lines.append(f'    if (str == "{lower_name}" || str == "tensor({lower_name})") return DataType::{cpp_name};')
    
    cpp_lines.extend([
        '    ',
        '    // Backward compatibility aliases',
        '    if (str == "float32") return DataType::kFloat32;',
        '    if (str == "float64") return DataType::kFloat64;',
        '    if (str == "unknown") return DataType::kUnknown;',
        '    ',
        '    return DataType::kUndefined;',
        '}',
        '',
        'size_t GetDataTypeSize(DataType dtype) {',
        '    switch (dtype) {',
    ])
    
    size_map = {
        'kUndefined': (0, '  // Undefined'),
        'kFloat': 4, 'kInt32': 4, 'kUint32': 4,
        'kDouble': 8, 'kInt64': 8, 'kUint64': 8, 'kComplex64': 8,
        'kComplex128': 16,
        'kString': (0, '  // Variable size'),
        'kBool': 1, 'kUint8': 1, 'kInt8': 1,
        'kUint16': 2, 'kInt16': 2, 'kFloat16': 2, 'kBFloat16': 2,
    }
    
    for name in onnx_to_cpp.values():
        if 'Float8' in name:
            size_map[name] = 1
        elif 'Float4' in name or name in ['kUint4', 'kInt4']:
            size_map[name] = (0, '  // 4-bit packed')
        elif name in ['kUint2', 'kInt2']:
            size_map[name] = (0, '  // 2-bit packed')
    
    for value, cpp_name in sorted(value_to_cpp.items()):
        val = size_map.get(cpp_name, 0)
        if isinstance(val, tuple):
            size, comment = val
        else:
            size = val
            comment = ''
        cpp_lines.append(f'        case DataType::{cpp_name}: return {size};{comment}')
    
    cpp_lines.extend([
        '        default: return 0;',
        '    }',
        '}',
        '',
        'std::string Dimension::ToString() const {',
        '    if (is_dynamic_) {',
        '        if (symbol_.empty()) return "?";',
        '        return symbol_;',
        '    }',
        '    return std::to_string(value_);',
        '}',
        '',
        'bool Dimension::operator==(const Dimension& other) const {',
        '    if (is_dynamic_ != other.is_dynamic_) return false;',
        '    if (is_dynamic_) return symbol_ == other.symbol_;',
        '    return value_ == other.value_;',
        '}',
        '',
        'Shape::Shape(std::initializer_list<int64_t> dims) {',
        '    for (int64_t d : dims) dims_.emplace_back(d);',
        '}',
        '',
        'bool Shape::IsDynamic() const {',
        '    for (const auto& dim : dims_) if (dim.IsDynamic()) return true;',
        '    return false;',
        '}',
        '',
        'std::optional<int64_t> Shape::GetTotalSize() const {',
        '    if (IsDynamic()) return std::nullopt;',
        '    int64_t total = 1;',
        '    for (const auto& dim : dims_) total *= dim.GetStaticValue();',
        '    return total;',
        '}',
        '',
        'std::string Shape::ToString() const {',
        '    std::ostringstream oss;',
        '    oss << "[";',
        '    for (size_t i = 0; i < dims_.size(); ++i) {',
        '        if (i > 0) oss << ", ";',
        '        oss << dims_[i].ToString();',
        '    }',
        '    oss << "]";',
        '    return oss.str();',
        '}',
        '',
        '}  // namespace oniris',
        '',
    ])
    
    # Write files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "types.hpp", 'w') as f:
        f.write('\n'.join(hpp_lines))
    
    with open(output_path / "types.cpp", 'w') as f:
        f.write('\n'.join(cpp_lines))
    
    print(f"\nGenerated files:")
    print(f"  - {output_path / 'types.hpp'}")
    print(f"  - {output_path / 'types.cpp'}")
    print(f"IR_VERSION: {ir_version}")
    print(f"Total data types: {len(dtypes)}")
    
    # Print new types
    new_types = [name for name, _ in dtypes 
                 if name in ['UINT2', 'INT2', 'FLOAT8E8M0', 'FLOAT4E2M1', 'UINT4', 'INT4']]
    if new_types:
        print(f"New/advanced types: {', '.join(sorted(new_types))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        proto_path = None
    else:
        proto_path = sys.argv[1]
    
    if len(sys.argv) < 3:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "src" / "core"
    else:
        output_dir = sys.argv[2]
    
    generate_dtype_from_proto(proto_path, output_dir)
