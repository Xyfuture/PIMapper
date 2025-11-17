"""
分析 benchmark 结果，计算 recursive_grid_search 相较于其他策略的性能提升
"""
import pandas as pd

df = pd.read_csv('results/benchmark_results_20251117_205640.csv')

for model in df['model_name'].unique():
    print(f"\n{model}:")
    for batch in sorted(df['batch_size'].unique()):
        data = df[(df['model_name'] == model) & (df['batch_size'] == batch)]

        rgs = data[data['strategy'] == 'recursive_grid_search']['total_latency'].values[0]
        h2llm = data[data['strategy'] == 'h2llm']['total_latency'].values[0]
        trivial = data[data['strategy'] == 'trivial']['total_latency'].values[0]

        improve_h2llm = (h2llm - rgs) / h2llm * 100
        improve_trivial = (trivial - rgs) / trivial * 100
        speedup_h2llm = h2llm / rgs
        speedup_trivial = trivial / rgs

        print(f"  batch={batch}: vs h2llm={improve_h2llm:.1f}% ({speedup_h2llm:.2f}x), vs trivial={improve_trivial:.1f}% ({speedup_trivial:.2f}x)")
