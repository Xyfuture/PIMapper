重构一下这个项目中的计算图部分. 

首先对 @core/graph/ops/base.py 进行重构

- class OpMetadata 是不需要的, 不要保留他
- class Op 需要进行大的改动
  - __init__ 方法中不需要 meta_data 相关的内容
  - to_dict 和 from_dict 要求删掉
  - can_convert_from_torch 和 convert_from 可以保留
  - 需要在类中内置一个 input_ops 的变量
    - 这个变量的类型是 OrderedDict 是一个有序的
    - key 的类型也是 class Op, 然后 value 的类型是一个 TensorSlice 的, 这类还没有定义, 下面会告诉你怎么定义, 这里表示 从那个 op 中来, 然后具体是这个 op 输出详细内容的中哪一部分(尤其是针对一个 op 有多个输出, tensor 还需要 slice 的情况)
  - 