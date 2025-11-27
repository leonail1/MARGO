# MARGO 图布局优化模块

## 概述

本模块实现了MARGO论文中的图布局优化算法，用于优化基于图的近似最近邻搜索索引的磁盘布局。通过将图节点分配到磁盘页面中，使得同一页面内的边权重和最大化，跨页面的边权重和最小化（最小割），从而减少磁盘随机I/O，提高搜索性能。

## 完整函数调用链

### 第一阶段：索引构建（myindex.cpp）

```
main() [构建索引的主程序]
  └─> myIndex::mybuild()
      ├─> copy_aligned_data_from_file()  // 加载向量数据
      └─> myIndex::my_build_with_data_populated()
          ├─> generate_frozen_point()  // 生成冻结点
          ├─> myIndex::mylink()  // 构建图结构
          │   └─> [多轮迭代]
          │       ├─> get_expanded_nodes()  // 搜索候选邻居
          │       ├─> myIndex::my_prune_neighbors()  // 剪枝
          │       │   └─> myIndex::my_occlude_list()  // RNG遮挡列表剪枝
          │       │       └─> [计算边权重 weight = 1.0 + 遮挡点数]
          │       └─> my_batch_inter_insert()  // 反向插入
          ├─> [计算入度] wws_[nbr] += 1.0
          ├─> [更新边权重] weight *= wws_[i]
          └─> mysave()  // 保存索引
              ├─> save_graph()  // 保存邻接表
              └─> my_save_weights()  // 保存边权重
```

### 第二阶段：图布局优化（new_mincut.cpp）

```
main() [new_mincut程序]
  └─> mincut::partition()
      ├─> mincut::my_kmeans()  // K-means聚类（256个簇）
      │   ├─> 采样训练聚类中心
      │   └─> 为所有节点分配簇标签
      │
      ├─> [构建簇倒排索引] cluster_ivf
      │
      ├─> [对每个簇并行处理]
      │   └─> cluster::cluster()  // 创建簇对象
      │       ├─> cluster::get_direct_graph()  // 提取簇内有向边
      │       ├─> cluster::trans2undirect()  // 转换为无向图
      │       │   ├─> 构建反向边
      │       │   ├─> 合并双向边
      │       │   └─> [边权重相加]
      │       └─> cluster::greedy_layout()  // 簇内贪心布局
      │           ├─> 收集所有边
      │           ├─> 按权重降序排序
      │           └─> [贪心分配到页面]
      │               ├─> 优先处理权重大的边
      │               ├─> 将边的两端点放入同一页面
      │               ├─> 贪心扩展：选择连接权重和最大的节点
      │               └─> 页面满后，开始新页面
      │
      ├─> mincut::merge_unassigned()  // 合并未分配节点
      │   └─> 收集所有簇中未满页面的节点
      │
      ├─> cluster::greedy_layout(post_process)  // 后处理簇布局
      │
      ├─> mincut::concatenate_pages()  // 连接所有页面
      │   ├─> 构建全局id2p_映射（节点→页面）
      │   └─> 构建全局p2id_映射（页面→节点集合）
      │
      └─> mincut::save_partition()  // 保存分区结果
```

## 关键算法说明

### 1. 边权重计算

边权重反映了边在图中的"覆盖能力"，计算分为三个阶段：

#### 阶段1：RNG剪枝时计算基础权重
在 `my_occlude_list()` 函数中：
```cpp
weight = 1.0 + 该边遮挡的候选点数量
```

**遮挡判断**（L2/余弦距离）：
- 对于边 (location, iter)，检查它是否遮挡候选点 iter2
- 遮挡条件：`dist(location, iter2) / dist(iter, iter2) > alpha`
- 几何意义：如果 iter2 离 iter 太近，说明通过 iter 可以"间接到达" iter2
- 每遮挡一个点，边权重加1

#### 阶段2：乘以起点入度
在 `my_build_with_data_populated()` 函数中：
```cpp
// 计算每个节点的入度
for(i in nodes) {
  for(nbr in graph[i]) {
    wws_[nbr] += 1.0;  // nbr的入度加1
  }
}

// 边权重 = 基础权重 × 起点入度
for(i in nodes) {
  for(weight in weights_[i]) {
    weight *= wws_[i];
  }
}
```

#### 阶段3：转换为无向图时合并权重
在 `trans2undirect()` 函数中：
```cpp
// 双向边的权重相加
weight_undirected(u, v) = weight(u→v) + weight(v→u)
```

### 2. K-means粗分区

- 目的：将相似的向量聚集在一起，作为后续细粒度分区的基础
- 默认分为256个簇
- 使用采样数据训练聚类中心，然后为所有节点分配簇标签

### 3. 簇内贪心布局

核心思想：**优先将边权重大的两个节点放在同一页面**

算法流程：
1. 按边权重降序排列所有边
2. 优先处理权重大的边，将其端点放入同一页面
3. 贪心扩展：选择与当前页面连接权重和最大的节点加入
4. 重复直到页面满或无可添加节点

伪代码：
```python
for edge in sorted_edges:
    u, v = edge.endpoints
    if page_is_empty:
        # 新页面：添加边的两个端点
        add_to_page(u, v)
    else:
        # 扩展页面：选择与当前页面连接权重和最大的节点
        best_node = argmax(sum(weights to current_page))
        add_to_page(best_node)
```

### 4. 后处理未分配节点

- 收集所有簇中未能分配到满页面的节点
- 形成一个后处理簇，重新进行布局
- 确保所有节点都被分配到某个页面

### 5. 全局页面连接

- 将所有簇的页面连接成全局页面序列
- 构建两个映射关系：
  - `id2p_`：节点ID → 页面ID
  - `p2id_`：页面ID → 节点ID集合

## 数据流

```
1. 索引构建阶段：
   向量数据 → 图结构(邻接表) → 边权重计算 → 保存到磁盘
   
2. 图布局优化阶段：
   加载图+权重 → K-means分簇 → 簇内贪心布局 → 全局页面连接 → 保存分区

3. 边权重演变：
   初始: 1.0
   → RNG剪枝: weight = 1.0 + 遮挡点数
   → 乘以入度: weight *= wws_[i]
   → 转无向图: weight_uv = weight_u→v + weight_v→u
   → 贪心布局: 按权重降序处理边
```

## 复杂度分析

- **K-means聚类**: O(n × k × iter)
  - n: 节点数
  - k: 簇数（默认256）
  - iter: 迭代次数
  
- **簇内贪心布局**: O(m log m) per cluster
  - m: 簇内边数
  
- **总体**: O(n × k + m log m)

## 使用方法

### 编译

```bash
cd my_gp
mkdir build && cd build
cmake ..
make
```

### 运行

```bash
./new_mincut <index_path_prefix> <base_data_path>
```

**参数说明**：
- `index_path_prefix`: 索引文件路径前缀
- `base_data_path`: 基础向量数据路径（用于K-means聚类）

**示例**：
```bash
./new_mincut /data/index/sift1M /data/sift1M_base.fbin
```

## 输入文件

1. `<index_path_prefix>_mem.index` - 内存图索引（邻接表）
2. `<index_path_prefix>_mem.index.weights` - 边权重文件
3. `<index_path_prefix>_disk.index` - 磁盘索引元数据
4. `<base_data_path>` - 基础向量数据

## 输出文件

1. `<index_path_prefix>_partition.bin` - 节点到页面的分区映射

该分区文件后续由 `index_relayout` 工具根据分区结果重新布局磁盘索引。

## 最终目标

- ✅ 最大化页面内的边权重和（局部性）
- ✅ 最小化跨页面访问（最小割）
- ✅ 优化磁盘I/O性能，将随机访问转换为顺序访问

## 参考文献

MARGO: A Disk-based Graph Layout Optimization Framework for Approximate Nearest Neighbor Search
