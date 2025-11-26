# 切换到release目录并编译
cd ../release && make -j && \

# SIFT100K数据集配置
path_prefix="../index/sift100k_M64_R64_L125_B0.003/"  # 索引文件路径前缀
base_file="../data/sift100k/sift_learn.fbin"          # 基础数据集文件

# 执行最小割图分区算法
./my_gp/new_mincut ${path_prefix} ${base_file}

# 根据分区结果重新布局索引文件
./tests/utils/index_relayout ${path_prefix}_disk.index ${path_prefix}_partition.bin

# 备份原始beam search索引文件
if [ ! -f "${path_prefix}_disk_beam_search.index" ]; then
    mv ${path_prefix}_disk.index ${path_prefix}_disk_beam_search.index
fi

# 将重新布局后的索引文件设为新的磁盘索引
mv ${path_prefix}_partition_tmp.index ${path_prefix}_disk.index