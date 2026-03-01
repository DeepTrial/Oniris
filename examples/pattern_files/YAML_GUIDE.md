# YAML Pattern 文件格式

## 概述

YAML 格式的 Pattern 文件提供更简洁、易读的格式，同时去除了不必要的元数据字段。

## 格式对比

### JSON 格式（旧）
```json
{
  "name": "fusion_patterns",
  "version": "1.0.0",
  "metadata": {
    "author": "team",
    "created": "2024-01-15"
  },
  "patterns": [
    {
      "name": "ConvRelu",
      "pattern_string": "Conv(?, c0)\nRelu(c0, ?)",
      "description": "Conv + ReLU fusion",
      "category": "fusion",
      "version": "1.0.0",
      "author": "ml-team",
      "tags": ["fusion", "activation"],
      "enabled": true,
      "priority": 100
    }
  ]
}
```

### YAML 格式（新）
```yaml
patterns:
  - name: ConvRelu
    pattern: |
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion
    cat: fusion
    priority: 100
```

## YAML 格式说明

### 基本结构

```yaml
patterns:
  - name: PatternName        # Pattern 名称（必需）
    pattern: |               # Pattern 定义（必需）
      Op1(?, output1)
      Op2(output1, ?)
    desc: Description        # 描述（可选）
    cat: category            # 类别（可选）
    priority: 100            # 优先级（可选）
    enabled: true            # 是否启用（可选，默认 true）
```

### 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | ✓ | Pattern 名称 |
| `pattern` | ✓ | Pattern 定义，使用 `\|` 表示多行 |
| `desc` | - | 简短描述 |
| `cat` | - | 类别：fusion/optimization/quantization/custom |
| `priority` | - | 优先级，数字越大越优先 |
| `enabled` | - | 是否启用，默认 true |

### 类别（cat）取值

- `fusion` - 算子融合
- `optimization` - 优化
- `quantization` - 量化
- `custom` - 自定义

## 完整示例

```yaml
# Fusion Patterns
patterns:
  # Conv + ReLU
  - name: ConvRelu
    pattern: |
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion
    cat: fusion
    priority: 100

  # Conv + BN + ReLU
  - name: ConvBnRelu
    pattern: |
      Conv(?, c0)
      BatchNormalization(c0, bn0)
      Relu(bn0, ?)
    desc: Conv + BN + ReLU fusion
    cat: fusion
    priority: 95

  # Swish activation
  - name: Swish
    pattern: |
      Conv(?, c0)
      Sigmoid(c0, s0)
      Mul([s0, c0], ?)
    desc: Swish activation fusion
    cat: fusion
    priority: 80
```

## 使用方法

### 1. 加载 YAML 文件

```python
import oniris

# 创建 Pattern Manager
pm = oniris.PatternManager()

# 加载 YAML 文件
patterns = oniris.load_yaml_patterns('patterns/fusion.yaml')

# 注册到 Pattern Manager
for p in patterns:
    pm.register_pattern(p)
```

### 2. 使用 YamlPatternLoader

```python
# 创建 Loader
loader = oniris.YamlPatternLoader()

# 加载单个文件
loader.load('patterns/fusion.yaml')

# 或加载整个目录
loader.load_all('patterns/')

# 获取 PatternManager
pm = loader.get_manager()
```

### 3. 简化的导入函数

```python
pm = oniris.PatternManager()

# 一行导入
count = oniris.import_yaml_patterns(pm, 'patterns/fusion.yaml')
print(f'Imported {count} patterns')
```

### 4. 应用到编译器

```python
# 创建编译器
compiler = pm.create_compiler()

# 编译模型
result = compiler.compile_file('input.onnx', 'output.onnx')
```

## 优势

1. **简洁** - 去除了 author, version 等不必要的字段
2. **可读** - YAML 格式更易于阅读和编辑
3. **注释** - 支持 `#` 开头的注释
4. **多行** - 使用 `|` 方便地定义多行 pattern
5. **简洁字段名** - 使用 `desc`, `cat` 代替 `description`, `category`

## 文件组织

```
patterns/
├── fusion.yaml       # 融合 Pattern
├── optimization.yaml # 优化 Pattern
├── quantization.yaml # 量化 Pattern
└── custom.yaml       # 自定义 Pattern
```

## 注意事项

1. 缩进使用 2 个空格（推荐）
2. Pattern 定义使用 `|` 后换行，保持正确的缩进
3. 类别名称小写：fusion, optimization, quantization, custom
4. 不需要的字段可以直接省略
