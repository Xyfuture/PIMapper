#!/usr/bin/env python3
"""调试策略错误"""

import traceback
from pimapper.core.hwspec import PIMChannelSpec, AcceleratorSpec, Accelerator
from pimapper.core.matrixspec import MatrixShape, DataFormat, DataType
from pimapper.matrixmapper.strategy.h2llm_mapping import H2LLMTilingStrategy

# 创建加速器
channel_spec = PIMChannelSpec(
    compute_power=4.0,
    shared_bandwidth=12.8,
    memory_bandwidth=1.0
)
accel_spec = AcceleratorSpec(
    channel_count=8,
    channel_spec=channel_spec
)
accelerator = Accelerator.create_from_spec(accel_spec)

# 创建矩阵
data_format = DataFormat(
    input_dtype=DataType.FP16,
    output_dtype=DataType.FP16,
    weight_dtype=DataType.INT4
)
matrix = MatrixShape(
    rows=4096,
    cols=4096,
    batch_size=20,
    data_format=data_format
)

# 测试 H2-LLM 策略
print("测试 H2-LLM 策略...")
try:
    h2llm_strategy = H2LLMTilingStrategy(element_size=2.0)
    result = h2llm_strategy.find_optimal_mapping(matrix, accelerator)
    print(f"成功: {result}")
except Exception as e:
    print(f"错误: {e}")
    print("\n完整堆栈跟踪:")
    traceback.print_exc()
