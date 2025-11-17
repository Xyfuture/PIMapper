"""
分析小模型 benchmark 结果，计算 recursive_grid_search 相较于 h2llm 的性能提升
"""
import pandas as pd

df = pd.read_csv('results/benchmark_results_20251117_214108.csv')

for model in df['model_name'].unique():
    print(f"\n{model}:")
    for batch in sorted(df['batch_size'].unique()):
        data = df[(df['model_name'] == model) & (df['batch_size'] == batch)]

        rgs = data[data['strategy'] == 'recursive_grid_search']['total_latency'].values[0]
        h2llm = data[data['strategy'] == 'h2llm']['total_latency'].values[0]

        improve = (h2llm - rgs) / h2llm * 100
        speedup = h2llm / rgs

        print(f"  batch={batch}: vs h2llm={improve:.1f}% ({speedup:.2f}x)")
