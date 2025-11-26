/*
 * 基于最小割的图分区算法
 * 
 * 功能说明:
 * 本程序实现了基于最小割(Min-Cut)的图分区算法，用于优化磁盘索引的数据布局。
 * 目标是将图节点分配到不同的磁盘页面中，使得同一页面内的边权重和最大化，
 * 跨页面的边权重和最小化（最小割），从而减少磁盘随机I/O，提高搜索性能。
 * 
 * 核心算法流程:
 * 
 * 1. K-means粗分区 (my_kmeans)
 *    - 对向量数据进行K-means聚类，将图节点划分为256个初始簇
 *    - 使用采样数据训练，然后为所有节点分配簇标签
 *    - 目的：将相似的向量聚集在一起，作为后续细粒度分区的基础
 * 
 * 2. 簇内贪心布局 (greedy_layout)
 *    - 对每个簇内部，执行贪心算法将节点分配到页面
 *    - 核心思想：优先将边权重大的两个节点放在同一页面
 *    - 算法流程：
 *      a) 按边权重降序排列所有边
 *      b) 优先处理权重大的边，将其端点放入同一页面
 *      c) 贪心扩展：选择与当前页面连接权重和最大的节点加入
 *      d) 重复直到页面满或无可添加节点
 * 
 * 3. 处理未分配节点 (merge_unassigned)
 *    - 收集所有簇中未能分配到满页面的节点
 *    - 形成一个后处理簇，重新进行布局
 *    - 确保所有节点都被分配到某个页面
 * 
 * 4. 全局页面连接 (concatenate_pages)
 *    - 将所有簇的页面连接成全局页面序列
 *    - 构建节点到页面的映射关系(id2p_)
 *    - 构建页面到节点集合的映射关系(p2id_)
 * 
 * 5. 保存分区结果 (save_partition)
 *    - 将最终的节点到页面的映射保存到文件
 *    - 后续由index_relayout工具根据此分区结果重新布局磁盘索引
 * 
 * 最终目标:
 * - 最大化页面内的边权重和（局部性）
 * - 最小化跨页面访问（最小割）
 * - 优化磁盘I/O性能，将随机访问转换为顺序访问
 * 
 * 使用示例:
 * ./new_mincut <index_path_prefix> <base_data_path>
 * 
 * 输入文件:
 * - <index_path_prefix>_mem.index: 内存图索引（邻接表）
 * - <index_path_prefix>_mem.index.weights: 边权重文件
 * - <index_path_prefix>_disk.index: 磁盘索引元数据
 * - <base_data_path>: 基础向量数据（用于K-means聚类）
 * 
 * 输出文件:
 * - <index_path_prefix>_partition.bin: 节点到页面的分区映射
 */

#include <iostream>
#include <mincut.h>
#include <string>


int main( int argc, char *argv[]) {

    // 检查命令行参数数量
    if( argc != 3) {
        std::cerr << "wrong input parameters" << std::endl;
        return 0;
    }
    
    // 定义各种文件路径
    std::string index_path;       // 内存索引路径
    std::string weight_path;      // 权重文件路径
    std::string disk_path;        // 磁盘索引路径
    std::string partition_path;   // 分区结果路径
    std::string reverse_path;     // 反向图路径
    std::string undirect_path;    // 无向图路径
    std::string edge_path;        // 边文件路径
    
    // 根据第一个参数构建各种文件的完整路径
    index_path = (std::string)argv[1] + "_mem.index";
    weight_path = (std::string)argv[1] + "_mem.index.weights";
    disk_path = (std::string)argv[1] + "_disk.index";
    partition_path = (std::string)argv[1] + "_partition.bin";
    reverse_path = (std::string)argv[1] + "_reverse.index";
    undirect_path = (std::string)argv[1] +"_undirect.index";
    edge_path = (std::string)argv[1] + "_sorted.edge";

    // 基础数据集路径（第二个参数）
    std::string base_path;
    base_path = (std::string)argv[2];

    // 记录开始时间
    auto begin_time = std::chrono::high_resolution_clock::now();
    
    // 创建最小割分区对象
    mincut *instance = new mincut();

    // 加载磁盘索引的元数据（节点数、维度等）
    instance->load_meta_data( disk_path);
    
    // 加载内存中的图结构（邻接表）
    instance->load_index_graph( index_path);
    
    // 执行图分区算法，将图划分为256个分区
    instance->partition( base_path, 256);
    
    // 保存分区结果到文件
    instance->save_partition( partition_path);

    // 计算并输出总耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( end_time - begin_time);
    std::cout << "mincut partition cost " << duration.count() << "s" << std::endl;
    
    // 释放内存
    delete instance;

    return 0;
}