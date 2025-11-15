# Archive Test Files

这个目录包含了 PIMapper 项目的测试文件归档。

## 运行测试

由于测试文件使用相对导入 `from pimapper.xxx`，需要从项目根目录运行测试，并确保项目根目录在 PYTHONPATH 中：

### Windows (PowerShell)
```powershell
# 设置 PYTHONPATH 并运行测试
$env:PYTHONPATH="D:\code\PIMapper"; python archive_test/test_full_pipeline.py
```

### Windows (CMD)
```cmd
# 设置 PYTHONPATH 并运行测试
set PYTHONPATH=D:\code\PIMapper && python archive_test\test_full_pipeline.py
```

### Linux/Mac
```bash
# 设置 PYTHONPATH 并运行测试
PYTHONPATH=/path/to/PIMapper python archive_test/test_full_pipeline.py
```

### 或者使用 Python -m 方式
```bash
# 从项目根目录运行
python -m archive_test.test_full_pipeline
```

## 测试文件列表

- `test_batched_matmul.py` - 批量矩阵乘法测试
- `test_ffn_matrix_mapping.py` - FFN 矩阵映射测试
- `test_full_pipeline.py` - 完整流程测试（模型追踪、归一化、简化）
- `test_fusion_matrix_mapping.py` - 融合矩阵映射测试
- `test_latency_calculation.py` - 延迟计算测试
- `test_llama_layer_latency.py` - LLaMA 层延迟测试
- `test_matrix_allocation_tree.py` - 矩阵分配树测试
- `test_matrix_fusion.py` - 矩阵融合测试
- `test_matrix_mapping_pass.py` - 矩阵映射 Pass 测试
- `test_recursive_grid_search_with_tree.py` - 递归网格搜索测试
- `test_simulator.py` - 模拟器测试
- `test_strategy_comparison.py` - 策略对比测试
- `test_tile_id_hierarchy.py` - Tile ID 层次结构测试
- `test_vector_latency.py` - 向量延迟测试
