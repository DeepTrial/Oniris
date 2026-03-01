# Pattern Manager

统一的用户 Pattern 管理系统，用于集中管理、分类、查询和导出导入 Pattern。

## 概述

Pattern Manager 提供了完整的 Pattern 生命周期管理：

- **注册与注销** - 添加、删除 Pattern
- **分类管理** - 按类别组织 Pattern（融合、优化、量化等）
- **状态控制** - 启用/禁用 Pattern
- **查询检索** - 按类别、标签、名称等条件查询
- **导入导出** - JSON 格式的 Pattern 序列化
- **全局注册表** - 应用级别的 Pattern 共享

## 快速开始

```python
import oniris

# 创建 Pattern Manager
pm = oniris.PatternManager()

# 注册 Pattern
pm.register_pattern(
    "ConvRelu",
    "Conv(?, c0)\nRelu(c0, ?)",
    oniris.PatternCategory.FUSION,
    "Conv + ReLU fusion"
)

# 查看统计
pm.print_summary()

# 应用到编译器
compiler = pm.create_compiler()
```

## Pattern 分类

| 类别 | 说明 | 示例 |
|------|------|------|
| `FUSION` | 算子融合模式 | Conv+ReLU, Conv+BN+ReLU |
| `OPTIMIZATION` | 优化模式 | 死代码消除，常量折叠 |
| `QUANTIZATION` | 量化相关 | QConv, QMatMul |
| `ANALYSIS` | 分析模式 | 性能分析，瓶颈检测 |
| `CUSTOM` | 用户自定义 | 任意用户定义模式 |

## API 参考

### PatternManager

#### 注册 Pattern

```python
# 简单注册
pm.register_pattern(
    name="ConvRelu",
    pattern_string="Conv(?, c0)\nRelu(c0, ?)",
    category=oniris.PatternCategory.FUSION,
    description="Conv + ReLU fusion"
)

# 使用 ManagedPattern 对象
pattern = oniris.ManagedPattern("ConvRelu", "Conv(?, c0)\nRelu(c0, ?)")
pattern.metadata.description = "Conv + ReLU"
pattern.metadata.tags = ["fusion", "activation"]
pattern.metadata.priority = 10
pm.register_pattern(pattern)
```

#### Pattern 查询

```python
# 获取所有 Pattern 名称
names = pm.get_pattern_names()

# 检查 Pattern 是否存在
exists = pm.has_pattern("ConvRelu")

# 获取统计信息
stats = pm.get_statistics()
print(f"Total: {stats.total_patterns}")
print(f"Enabled: {stats.enabled_patterns}")
print(f"By category: {stats.category_counts}")

# 按类别获取
fusion_patterns = pm.get_patterns_by_category(oniris.PatternCategory.FUSION)

# 按标签获取
tagged_patterns = pm.get_patterns_by_tag("fusion")
```

#### Pattern 状态管理

```python
# 启用/禁用单个 Pattern
pm.set_pattern_enabled("ConvRelu", False)
is_enabled = pm.is_pattern_enabled("ConvRelu")

# 按类别启用/禁用
pm.set_category_enabled(oniris.PatternCategory.FUSION, False)

# 获取启用数量
enabled_count = pm.get_enabled_pattern_count()
```

#### 导入导出

```python
# 导出为 JSON
json_str = pm.export_to_json(pretty=True)
with open("patterns.json", "w") as f:
    f.write(json_str)

# 从文件导出
pm.export_to_file("patterns.json", pretty=True)

# 从 JSON 导入
with open("patterns.json") as f:
    count = pm.import_patterns_from_json(f.read())

# 从文件导入
count = pm.import_patterns_from_file("patterns.json")
```

#### 批量导入 Collection

```python
# 导入内置集合
fusion = oniris.get_fusion_patterns()
pm.import_patterns(fusion)

opt = oniris.get_optimization_patterns()
pm.import_patterns(opt)

quant = oniris.get_quantization_patterns()
pm.import_patterns(quant)
```

### 全局 Pattern Registry

全局单例模式，用于应用级 Pattern 共享：

```python
# 获取全局注册表
registry = oniris.get_pattern_registry()

# 注册 Pattern（全局可用）
registry.register_pattern("MyPattern", "Conv(?, c0)\nRelu(c0, ?)")

# 获取全局 Manager
manager = registry.get_manager()
print(f"Global patterns: {manager.get_pattern_count()}")

# 加载内置 Pattern
registry.load_builtin_patterns()
```

### 与 Compiler 集成

```python
# 方法 1: 直接创建编译器
pm = oniris.PatternManager()
pm.import_patterns(oniris.get_fusion_patterns())
compiler = pm.create_compiler()

# 方法 2: 应用到现有编译器
compiler = oniris.ModelCompiler()
pm.apply_to_compiler(compiler)

# 获取 PatternDefinitions 列表
definitions = pm.get_enabled_pattern_definitions()
```

## 内置 Pattern Collections

### 融合模式 (Fusion Patterns)

```python
fusion = oniris.get_fusion_patterns()
# 包含:
# - ConvRelu: Conv + ReLU
# - ConvBnRelu: Conv + BatchNorm + ReLU
# - ConvBn: Conv + BatchNorm
# - GemmRelu: Gemm + ReLU
```

### 优化模式 (Optimization Patterns)

```python
opt = oniris.get_optimization_patterns()
# 包含:
# - Identity: Identity 消除
# - ReshapeReshape: 连续 Reshape 合并
```

### 量化模式 (Quantization Patterns)

```python
quant = oniris.get_quantization_patterns()
# 包含:
# - QConv: 量化 Conv 模式
```

### 所有内置模式

```python
all_patterns = oniris.get_all_builtin_pattern_collections()
```

## Pattern Metadata

每个 Pattern 都有丰富的元数据：

```python
pattern = oniris.ManagedPattern("MyPattern", "Conv(?, c0)\nRelu(c0, ?)")

# 基础信息
pattern.metadata.name = "MyPattern"
pattern.metadata.description = "Custom fusion pattern"
pattern.metadata.author = "user@example.com"
pattern.metadata.version = "1.0.0"

# 分类和标签
pattern.metadata.category = oniris.PatternCategory.CUSTOM
pattern.metadata.tags = ["fusion", "custom", "experimental"]

# 优先级（数字越大优先级越高）
pattern.metadata.priority = 100

# 启用状态
pattern.metadata.enabled = True

# 自定义属性
pattern.metadata.attributes["platform"] = "arm"
pattern.metadata.attributes["speedup"] = "1.5x"
```

## JSON 格式

Pattern Collection 的 JSON 格式：

```json
{
  "name": "my_patterns",
  "description": "Custom pattern collection",
  "version": "1.0.0",
  "metadata": {
    "author": "user",
    "created": "2024-01-15"
  },
  "patterns": [
    {
      "name": "ConvRelu",
      "pattern_string": "Conv(?, c0)\nRelu(c0, ?)",
      "description": "Conv + ReLU fusion",
      "category": "fusion",
      "version": "1.0.0",
      "tags": ["fusion", "activation"],
      "enabled": true,
      "priority": 10
    }
  ]
}
```

## 完整示例

### 示例 1: 自定义 Pattern 库

```python
import oniris

# 创建 Pattern Manager
pm = oniris.PatternManager()

# 定义自定义 Pattern
my_patterns = [
    ("CustomConvRelu", "Conv(?, c0)\nRelu(c0, ?)", "Custom Conv+ReLU"),
    ("CustomResidual", "Conv(?, c0)\nAdd([c0, ?], ?)", "Residual block"),
]

for name, pattern_str, desc in my_patterns:
    pm.register_pattern(name, pattern_str, 
                       oniris.PatternCategory.CUSTOM, desc)

# 启用所有 Custom 类别的 Pattern
pm.set_category_enabled(oniris.PatternCategory.CUSTOM, True)

# 导出供团队使用
pm.export_to_file("my_patterns.json", pretty=True)
```

### 示例 2: Pattern 版本管理

```python
import oniris

pm = oniris.PatternManager()

# 注册 v1.0 Pattern
pattern_v1 = oniris.ManagedPattern("ConvFusion", "Conv(?, c0)\nRelu(c0, ?)")
pattern_v1.metadata.version = "1.0.0"
pm.register_pattern(pattern_v1)

# 更新到 v2.0（覆盖）
pattern_v2 = oniris.ManagedPattern("ConvFusion", 
    "Conv(?, c0)\nBatchNormalization(c0, bn0)\nRelu(bn0, ?)")
pattern_v2.metadata.version = "2.0.0"
pm.register_pattern(pattern_v2, overwrite=True)

# 查看当前版本
p = pm.get_pattern("ConvFusion")
print(f"Current version: {p.metadata.version}")
```

### 示例 3: 条件查询

```python
import oniris

pm = oniris.PatternManager()

# 导入所有内置模式
pm.import_patterns(oniris.get_all_builtin_pattern_collections())

# 构建查询
query = oniris.PatternQuery()
query.category = oniris.PatternCategory.FUSION
query.enabled_only = True
query.min_priority = 5
query.name_contains = "Conv"

# 执行查询
results = pm.query_patterns(query)
print(f"Found {len(results)} patterns matching criteria")
```

### 示例 4: 与 Compiler 集成

```python
import oniris

# 配置 Pattern Manager
pm = oniris.PatternManager()

# 只导入需要的 Pattern
fusion = oniris.get_fusion_patterns()
pm.import_patterns(fusion)

# 禁用不需要的 Pattern
pm.set_pattern_enabled("GemmRelu", False)

# 创建并配置编译器
compiler = pm.create_compiler()
options = oniris.CompilerOptions()
options.verbose = True

# 编译模型
result = compiler.compile_file("model.onnx", "optimized.onnx", options)
print(f"Pattern matches: {result.pattern_matching_summary.total_matches}")
```

## 最佳实践

### 1. Pattern 命名规范

```python
# 推荐格式: [OpType][Detail][Version]
pm.register_pattern("ConvBnReluV2", ...)
pm.register_pattern("TransformerAttention", ...)
```

### 2. 优先级设置

```python
# 高优先级 Pattern 先匹配
pm.set_pattern_priority("CriticalFusion", 100)
pm.set_pattern_priority("OptionalFusion", 10)
```

### 3. 分类使用

```python
# 根据用途选择合适的类别
pm.register_pattern(..., category=oniris.PatternCategory.FUSION)
pm.register_pattern(..., category=oniris.PatternCategory.QUANTIZATION)
```

### 4. 团队协作

```python
# 导出团队共享的 Pattern
team_patterns = pm.export_patterns("team_fusion_patterns")
team_patterns.metadata["team"] = "ml-platform"
team_patterns.metadata["reviewed"] = "true"
team_patterns.save_to_file("/shared/team_patterns.json")

# 团队成员导入
pm.import_patterns_from_file("/shared/team_patterns.json")
```

## 注意事项

1. **全局注册表是单例** - 全局 Pattern Registry 在应用生命周期内共享
2. **Pattern 名称唯一** - 同名 Pattern 注册时需要 `overwrite=True`
3. **内存管理** - Pattern Manager 管理 Pattern 生命周期
4. **线程安全** - Pattern Manager 的读操作是线程安全的

## 相关文档

- [COMPILER.md](COMPILER.md) - ONNX Model Compiler 文档
- [ONNX_MATCHER_STYLE.md](ONNX_MATCHER_STYLE.md) - Pattern 语法文档
