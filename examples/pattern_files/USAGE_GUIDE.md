# Pattern 文件管理使用指南 (YAML)

## 概述

本指南介绍如何使用 YAML 格式的 Pattern 文件来组织、管理和使用 Pattern。

## 目录结构

```
pattern_files/
├── patterns/                    # Pattern 文件目录
│   ├── fusion_patterns.yaml    # 融合 Pattern
│   ├── optimization_patterns.yaml  # 优化 Pattern
│   └── custom_patterns.yaml    # 自定义 Pattern
├── pattern_cli.py              # 命令行工具
├── example_usage.py            # 使用示例
└── complete_demo.py            # 完整演示
```

## YAML 格式

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
```

### 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | ✓ | Pattern 名称 |
| `pattern` | ✓ | Pattern 定义，使用 `\|` 表示多行 |
| `desc` | - | 简短描述 |
| `cat` | - | 类别：fusion/optimization/quantization/custom |
| `priority` | - | 优先级，数字越大越优先 |

### 类别（cat）取值

- `fusion` - 算子融合
- `optimization` - 优化
- `quantization` - 量化
- `custom` - 自定义

## 命令行工具

### 列出所有 Pattern 文件

```bash
python pattern_cli.py list
```

输出：
```
Pattern 文件目录: /path/to/patterns

----------------------------------------------------------------------

文件: fusion_patterns.yaml
  Pattern 数量: 5
  类别分布: {'fusion': 5}
...
```

### 查看特定 Pattern 文件

```bash
python pattern_cli.py show fusion
```

### 验证 Pattern 文件

```bash
# 验证所有文件
python pattern_cli.py validate

# 验证特定文件
python pattern_cli.py validate fusion
```

### 测试加载 Pattern 文件

```bash
python pattern_cli.py test fusion
```

## 在代码中使用

### 基本用法

```python
import oniris

# 创建 Pattern Manager
pm = oniris.PatternManager()

# 从 YAML 加载 Pattern
count = oniris.import_yaml_patterns(pm, 'patterns/fusion.yaml')
print(f'Loaded {count} patterns')

# 应用到编译器
compiler = pm.create_compiler()

# 编译模型
result = compiler.compile_file('input.onnx', 'output.onnx')
```

### 批量加载多个文件

```python
import os

pm = oniris.PatternManager()
pattern_dir = 'patterns'

for filename in os.listdir(pattern_dir):
    if filename.endswith('.yaml'):
        filepath = os.path.join(pattern_dir, filename)
        count = oniris.import_yaml_patterns(pm, filepath)
        print(f'{filename}: {count} patterns')
```

### 管理 Patterns

```python
# 禁用特定 Pattern
pm.set_pattern_enabled('ConvBnRelu', False)

# 按类别启用/禁用
pm.set_category_enabled(oniris.PatternCategory.OPTIMIZATION, False)

# 设置优先级
pm.set_pattern_priority('ConvRelu', 100)

# 查看统计
pm.print_summary()
```

## 完整工作流示例

```python
import oniris

# 1. 创建 Pattern Manager
pm = oniris.PatternManager()

# 2. 从 YAML 导入 Patterns
pm.import_yaml_patterns('patterns/fusion.yaml')
pm.import_yaml_patterns('patterns/custom.yaml')

# 3. 管理 Patterns
pm.set_pattern_enabled('GemmRelu', False)  # 禁用特定 Pattern
pm.set_pattern_priority('ConvRelu', 100)   # 设置高优先级

# 4. 创建编译器
compiler = pm.create_compiler()

# 5. 编译模型
options = oniris.CompilerOptions()
options.pattern_match_before_opt = True
result = compiler.compile_file('model.onnx', 'optimized.onnx', options)

# 6. 分析结果
print(f'Matches: {result.pattern_matching_summary.total_matches}')
print(result.to_json())
```

## 最佳实践

### 1. 按类别组织文件

- `fusion_patterns.yaml` - 算子融合
- `optimization_patterns.yaml` - 优化
- `quantization_patterns.yaml` - 量化
- `custom_patterns.yaml` - 自定义

### 2. 命名规范

```yaml
# 推荐格式: [OpType][Detail]
- name: ConvRelu
- name: ConvBnRelu
- name: TransformerAttention
```

### 3. 优先级设置

```python
# 关键 Pattern 设置高优先级
pm.set_pattern_priority('CriticalFusion', 100)

# 可选 Pattern 设置低优先级
pm.set_pattern_priority('OptionalFusion', 10)
```

### 4. 文档注释

```yaml
# Fusion Patterns
# 用于 CNN 网络的算子融合优化

patterns:
  # Conv + ReLU 融合
  # 常见于卷积层后接激活函数
  - name: ConvRelu
    pattern: |
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion
    cat: fusion
    priority: 100
```

## 故障排除

### Pattern 无法加载

1. 验证 YAML 格式：`python pattern_cli.py validate`
2. 检查必需字段：name, pattern
3. 确保缩进正确（2 个空格）

### Pattern 匹配失败

1. 确认 Pattern 已启用：`pm.is_pattern_enabled('PatternName')`
2. 验证 Pattern 语法：`pattern.is_valid()`
3. 检查匹配时机：`options.pattern_match_before_opt`

## 参考

- [YAML 格式说明](YAML_GUIDE.md)
- [Pattern Manager 文档](../../docs/PATTERN_MANAGER.md)
- [Model Compiler 文档](../../docs/COMPILER.md)
