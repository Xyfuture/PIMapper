你要对 @pimapper/matrixmapper/strategy/recursive_grid_search.py 进行重构, 现在的算法在搜索中不会指定具体的分配标号, 例如我们只知道一个 channel 分到了那些 tile, 每个tile 是什么形状的, 但是不知道每个 tile 具体对应在原始矩阵中的位置. 本次重构的目的是建立一个 tree , 在实现好具体的分配策略, 能够在知道每个 channel 具体是分到了哪一块tile. 

---

首先我们来看 tree 的构建方式. 

在一次递归中, 对于一个矩阵, 我们按照 grid 对这个矩阵进行划分后, 会得到很多的 tile , 这些 tile 按照 round-robin 的方式首先分配到了 channel中, 然后还剩下了 余数个 tile 无法均匀分给所有的 channel, 此时采取了 递归的策略, 也就是将剩下的 tile 聚合成 1 个或者 2 个新的矩阵, 将新的矩阵视作一个子问题,进行递归. 

我们可以将划分出来的 tile 分成两类, 一类是 channel tiles, 表示这些 tile 已经被划分到了 channel , 余数的 tile 我们称之为 tail tiles, 这些 tile 要被作为子问题进行递归.  我们可以在划分后, 按照行优先的顺序进行编号, 每一个 tile 都有一个自己的编号. 

tree 中的节点也分为两类, 一类是 internal node ,一类是 leaf node . leaf node 表示的是 channel tiles, 这个节点包含了所有的 channel tiles 的编号, 已经每个 tile 被映射到了那个 channel 中, 同理也可以知道这个节点下, 每个 channel 需要运算几个 tile. internal node 则表示一个待分配的矩阵, 这个矩阵有一个基本的信息是 rows x cols 表示的 shape. 对于非 root node 的 internal node, 它来源于 tail tiles 拼接而成的, 那么就要保存 它是由那些 tiles 编号 拼出来的, 还要保留拼接是按照 m x n 的信息.  每个 internal node 都是一个矩阵, 因此还要被分割, 所以, 我们还要记录对于 internal node 的分割方式(grid 的信息). internal node 下面的 children 编号都是新的, 依赖于 internal node 的分割方式.

如此一来, 我们就能知道具体每个 channel 执行了那个 tile, 这个 tile 在矩阵中是什么位置了.

按照目前的分割方式, 每个 internal node 最多有 3 个节点, 一个 leaf node + 两个 children internal node, 只有一个 leaf node 也是合法的, 一个 leaf node + 一个 internal node 也是合法的, 没有 leaf node 不合法. 

做一个检查函数, 检查这个 tree 是否合法, 自顶向下进行. 

---

再接下来, 我们来看如何为 channel tiles 和 tail tiles 分配编号. 我们已经按照行优先的顺序进行了编号, 每一个 tile 都有一个自己的编号.  那么具体那些编号的 tile 会被认定为 channel tiles, 那些标号的 tile 会被认定为 tail tiles 呢.

我们要对 tree 进行两次遍历才能完成的构建出这棵树, 并完成序号分配工作.

第一次遍历主要是构建, 我们不会具体的分配编号, 而是知道 leaf node 中有几个 tile, 每个 internal node 是几个 tile 按照 什么 m x n 组成的. 

第二次遍历则要进行编号分配工作. 我们要自root 向leaf 进行. 首先检查 internal node 的 children, 首先对对于 children中的internal node 进行分配, 因为其有形状的要求, 我们要保证分配好的编号能确实组成他对应的形状. 然后对leaf node进行分配, 从剩下的tile 中 按照 round robin 的方式分配就可以了. 

---

你考虑上面的实现方式后, 考虑一下那些通用的函数可以抽象出来. 然后在 @pimapper/matrixmapper/ 中创建一个 utils 文件夹, 将大部分关于tree的代码放置到 matrix_allocation_tree.py 中. 





