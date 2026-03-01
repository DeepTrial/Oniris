# Pattern 文件管理 (YAML)

本目录包含 YAML 格式的 Pattern 文件管理示例和工具。

## 目录结构

```
pattern_files/
├── README.md                    # 本文件
├── YAML_GUIDE.md               # YAML 格式详细说明
├── patterns/                    # Pattern 文件目录
│   ├── fusion_patterns.yaml    # 融合 Pattern
│   ├── optimization_patterns.yaml  # 优化 Pattern
│   └── custom_patterns.yaml    # 自定义 Pattern
├── pattern_cli.py              # 命令行工具
├── example_usage.py            # 使用示例
├── complete_demo.py            # 完整演示
└── yaml_minimal_example.py     # 极简示例
```

## YAML 格式特点

- **简洁** - 只保留必要字段（name, pattern, desc, cat, priority）
- **易读** - YAML 语法更友好
- **注释** - 支持 `#` 注释
- **多行** - 使用 `|` 定义多行 pattern

## 示例 YAML 文件

```yaml
patterns:
  - name: ConvRelu
    pattern: |
      Conv(?, c0)
      Relu(c0, ?)
    desc: Conv + ReLU fusion
    cat: fusion
    priority: 100

  - name: ConvBnRelu
    pattern: |
      Conv(?, c0)
      BatchNormalization(c0, bn0)
      Relu(bn0, ?)
    desc: Conv + BN + ReLU fusion
    cat: fusion
    priority: 95
```

## 快速开始

### 列出所有 Pattern 文件

```bash
python pattern_cli.py list
```

### 查看特定 Pattern 文件

```bash
python pattern_cli.py show fusion
```

### 验证 Pattern 文件

```bash
python pattern_cli.py validate
```

### 测试加载 Pattern 文件

```bash
python pattern_cli.py test fusion
```

## 在代码中使用

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

## 字段说明

| 字段 | 必需 | 说明 |
|------|------|------|
| `name` | ✓ | Pattern 名称 |
| `pattern` | ✓ | Pattern 定义，使用 `\|` 多行 |
| `desc` | - | 描述 |
| `cat` | - | 类别：fusion/optimization/quantization/custom |
| `priority` | - | 优先级（数字越大越高） |

## 运行示例

```bash
# 极简示例
python yaml_minimal_example.py

# 完整示例
python example_usage.py

# 完整演示
python complete_demo.py
```

## 参考

- [YAML_GUIDE.md](YAML_GUIDE.md) - YAML 格式详细说明
- [Pattern Manager 文档](../../docs/PATTERN_MANAGER.md)
- [Model Compiler 文档](../../docs/COMPILER.md)
