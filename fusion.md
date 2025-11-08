做一个精通 ML 编译的专家, 你要帮我实现一个复杂的 pass, 这个 pass 已经有了雏形, 但是问题太多, 请你帮我重构这个 pass, 完整的实现功能. 

这个 pass 是用来实现矩阵融合操作的, 多个共享同一个输入的矩阵可以进行融合操作, 合并为一个大的矩阵. 但是融合的时候需要决定矩阵的Order. 我设计了两种 Order 的顺序, 一个是按照指定的顺序将多个矩阵排布, 直接规定好谁在前, 谁在后; 另一个是将多个矩阵的权重交错分布, 例如 矩阵 A 有 n 个 block, 矩阵 B 有 m 个 block, 则每 n/m 个 A 的 Block 之后 拼接一个 B 的 Block, 注意这个 Block 是沿着列的维度对矩阵进行的拆分, Block 之间不需要进行 reduce 操作. 

关于 Order 策略的选择,  有一个简单的算法, 就是在计算图上跑一个最晚开工时间的算法, 这样每个节点都有了一个最晚开工时间的 last_start_tag. 在最晚开工时间的算法中, 我们规定矩阵运算的时间是 1, 其他操作的时间假设是 0. 我们找到需要合并的矩阵之后, 检查这多个矩阵的 last_start_tag, 如果多个矩阵的 last_start_tag 是相同的, 那么就按照 交错的方式排布这几个矩阵, 如果几个矩阵的last_start_tag 是有先后顺序的, 那么让 应该最早开始的 矩阵排在最前面. 注意我们要处理的是多个矩阵的情况, 因此这个关系要组织成一个树的结构, 叶节点是需要合并的多个矩阵, parent 节点则说明多个 children 节点应该按照 哪一种策略合并. 最终到根节点, 得到最终的融合后的矩阵的Order 情况. 

这棵树从下向上走可以得到融合后矩阵的形状, 而从上向下走则能得到输出的情况. 原先的每个矩阵的输出都要被后面的节点使用, 但是合并之后, 得到的输出是汇聚了好几个矩阵结果的, 我们还必须通过这棵树的信息从 合并矩阵的结果 中提取出对应矩阵的结果, 用于构建合并后的计算图.

 现在请你实现这个功能, 按照我下面的指导:

对 @pimapper/core/graph/ops/fusionmatrix.py 进行彻底的重构:

- 使 FusionMatMulOp 这个 class 能够满足 @pimapper/graph/ops/base.py 中的 op 定义, 把相关的部分都补全. 记录好新的矩阵的维度信息
- 在这个文件中定义好两种 Order 的策略, 同时构建一个 合并树 的结构, 叶节点为原始的 Op (注意在 pass 中删除 原始 Op 的连接关系), parent 节点记录策略. 要注意处理 交错模式下, 交错的粒度. 

对 @pimapper/modelmapper/passes/matrix_fusion.py 进行彻底的重构, 实现 找可聚合 op, 替换为 FusionMatMulOp 的过程

- 适应好  @pimapper/graph/ops/base.py 中对 Op 的定义, 能够处理好 input_ops 和 results 的情况
- 实现好 最晚开工时间的算法, 能给每个 op 一个 tag
- 能正确找到需要合并的矩阵,
- 每次合并中, 能正确处理 输入输出的关系
  - 要能删除 原始 Op 在图中连接关系
  - 能正确处理好, fusionMatMulOp 的 results 和后续 op 连接的情况的, 需要使用 合并树中的信息 通过级联(可选) @pimapper/core/ops/base.py 中的 TensorTransform 来筛选出 每个 后续 Op 真正需要的输入, 这是一个难点, 请你仔细思考怎么处理
  - 能维护好 所有的 shape 的关系, 矩阵的尺寸, 输入输出的尺寸等等







你这个实现还是存在问题的, 我期望的是从root 到 leaf 的一个搜索, 每经过一个internal node 就要在 ResultFilter 中加一个 TensorTransform, 例如一个 InternalNode 在他的第 2 个 children 中找到了需要的输出, 就要加一个 filter , 从 InternalNode 的输出中提取出第2 个 Children 的结果的位置, 如果是sequential 的 , 就是直接切片出对应的结果, 用[start:end] 的形式就可以了, 如果是 interleaved , 那么要确定好 stride, 每隔多少个提取多少个(可能需要一个新的TensorTransform 变换). 最后这个 Filter 的熟悉应该注意, 每次都把 靠近 Root 的 Node 的 Filter 放前面. ResultFilter 应该在 Root 中创建