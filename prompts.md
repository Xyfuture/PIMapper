读一下 @pimapper/modelmapper/converter.py 中的 summarize_graph 函数, 将这个函数重构到 @pimapper/core/graph/base.py 中, 现在要求支持如下的几个功能:

- 依次输出每个节点
- 对于每个节点, 要输出如下的信息
  - print 该节点的所有输入 ops, 每个输入还是对应输出 result filter 的信息, 最好能告知 输入的尺寸
  - 如果这个节点有一些内部信息, 例如矩阵尺寸之类的, 就要输出相关信息

实现完毕后, 将这个 改动重新同步到所有用到的文件之中 . 



---



重构一下 matrixmapper 的部分. matrix mapper 的主要工作是将一个矩阵映射到多个 PIM channel 中. 现在有 3 种映射算法, 一个是 从 h2-llm 论文中提取出来的算法 在 @pimapper/matrixmapper/strategy/h2llm_mapping.py 中, 另一个是最简单的算法, 直接按照指定的形状, 将矩阵分成多个 tile ,然后顺序分配给多个 channel 来运算, 相关代码在 @pimapper/matrixmapper/strategy/trivial.py 中, 最后一个是我在我的论文要使用的映射算法, 在 @pimapper/matrixmapper/strategy/agent_grid_search.py 中, 其通过递归搜索的方式来确定最优的策略. 

现在要求你简单重构一下代码, 精简一下代码, 实现一些功能:

- 对 @pimapper/core/hwspec.py 中的名字进行一下更改 

  - ComputeDie 更改为 PIMChannel , Chip 的定位现在是 Accelerator, 也要进行改名, 添加一个额外的 Host class 来描述 与 PIMChannel 进行连接
  - 现在的层级是这样的
    - Accelerator
      - Host
      - PIMChannel

- 将 @pimapper/sim/sim_engine.py 迁移到 @pimapper/matrixmapper/ 目录下面, 因为这个 sim_engine 是最后要使用的 sim_engine, 这个只是 matrixmapper 内部使用的一个简单的评估器, 你最好将 sim_engine 的 class 的名字进行修改, 因为这个只在 matrixmapper 内部使用, 要和外面的区分开

  





- 支持生成能用的分配
- 方便之后的地址分配



---

MatrixModel 的任务

- [ ] 生成可用的分块, 考虑到后续的地址分配
- [ ] 给出一个调用的接口