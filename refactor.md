重构一下这个项目中的计算图部分. 

首先对 @pimapper/core/graph/ops/base.py 进行重构

- class OpMetadata 是不需要的, 不要保留他
- 添加一个 class GraphTensor
  - 内部用于记录一个 Tensor 的维度信息, 不需要存储记录实际的值
- 添加一个 class GraphTensorSlice 
  - 内部记录如何对一个 GraphTensor 进行的 slice 操作, 只需要记录信息即可(slice的维度等信息, 能够描述出如何进行 slice), 不需要进行实际的操作. 
- 添加一个 class ResultFilter 类
  - 对一个 Op 内部的 results:list[GraphTensor ]成员进行筛选, 主要有两个参数, 一个用来定位是results 这个 list 中的第几个, 另一个参数用来记录对这个 Tensor进行的 TensorSlice 操作(可以为 None) 
- class Op 需要进行大的改动
  - __init__ 方法中不需要 meta_data 相关的内容
  - to_dict 和 from_dict 要求删掉
  - can_convert_from_torch 这个方法删除
  - convert_from_torch 保留, 但要进行改动
    - 输入应该是一个 @pimapper/core/graph/ops/torch_compat.py 中的 TorchFxOp, 这里不需要具体的实现
  - 需要在类中内置一个 input_ops 的变量
    - 这个变量的类型是 OrderedDict 是一个有序的
    - key 的类型也是 class Op, 然后 value 的类型是一个 ResultFilter的, 表示从那个 op 中来, 然后具体是这个 op 输出详细内容的中哪一部分(尤其是针对一个 op 有多个输出, tensor 还需要 slice 的情况)
  - 在类中放置一个 results 的 list, 类型是 GraphTensor 类型的
  - 删除 get_input_refs 这个方法
  - 输入和输出都是能随后指定和新加的, 不强制在 __init__.py 中就进行初始化

随后对 @pimapper/core/graph/ops/native.py 进行重构, 首先是同步上面的改动, 然后执行如下的工作:

- 更新 convert_from_torch 操作, 判断一下能不能对应的 Torch 能不能转换成自己, 不能的话报错, 能的话进行转换操作.
- __init__.py 中都不要设置 输入op 的信息了
- 整个文件简单一些
- 属于 op 自己的信息还是要保留的, 例如 matmul 中矩阵的维度

然后对 @pimapper/modelmapper/converter.py 进行改动

- 删除其中的simplify_computation_graph函数, 因为后面通过 passes 中的 simplify.py 写了一个 pass 替换掉了
- 要能同步一下 input_ops 和 results 的改动, 把 Tensor 的 dim 信息记录下来, 目前所有的 results 都设置为 1 个, 然后 resultfilter 都是取默认的 result 就可以了, 也不需要进行 slice 操作, 这里主要是对 fx_to_computation_graph 中的一些修改. 

对 @pimmapper/modelmapper/passes/simplify.py 要同步改动的影响, simplify 的时候主要维持 input_ops 和 results 的关系

再之后同步对 @pimapper/modelmapper/passes/normalize_ops.py 进行改动, 主要是将上面的改动同步过来, 核心的逻辑不需要变化

对于其他的文件, 如果被这次的改动影响到了, 也要进行同步的修改

