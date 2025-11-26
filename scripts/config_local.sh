#!/bin/sh
# 加载数据集配置
source config_dataset.sh

# 通过取消下面行的注释来选择数据集
# 如果多行被取消注释,则只有最后一个数据集生效
dataset_SIFT100K

##################
#   Disk Build   #
#   磁盘索引构建  #
##################
R=64           # 图的最大度数
BUILD_L=125    # 构建时的搜索列表大小
M=64           # 每次构建的批次大小
BUILD_T=8      # 构建时的线程数

##################
#       SQ       #
#     标量量化    #
##################
USE_SQ=0       # 是否使用标量量化(0=否, 1=是)

##################################
#   In-Memory Navigation Graph   #
#      内存导航图配置             #
##################################
MEM_R=64                    # 内存图的最大度数
MEM_BUILD_L=125             # 内存图构建时的搜索列表大小
MEM_ALPHA=1.2               # Alpha参数(用于图构建)
MEM_RAND_SAMPLING_RATE=0.1  # 随机采样率
MEM_USE_FREQ=0              # 是否使用频率文件(0=否, 1=是)
# MEM_FREQ_USE_RATE=0.01    # 频率使用率(当MEM_USE_FREQ=1时)

##########################
#   Generate Frequency   #
#      生成频率文件       #
##########################
# FREQ_QUERY_FILE=$QUERY_FILE  # 频率查询文件路径
# FREQ_QUERY_CNT=0             # 查询数量(设置为0表示使用全部,默认)
# FREQ_BM=4                    # Beam宽度
# FREQ_L=100                   # 搜索列表大小(目前只支持单个值)
# FREQ_T=16                    # 线程数
# FREQ_CACHE=0                 # 缓存节点数
# FREQ_MEM_L=0                 # 内存搜索列表大小(非零启用)
# FREQ_MEM_TOPK=10             # 内存TopK

#######################
#   Graph Partition   #
#      图分区配置      #
#######################
GP_TIMES=16      # 分区数量
GP_T=112         # 图分区的线程数
GP_LOCK_NUMS=0   # 初始化时锁定的节点数, lock_node_nums = partition_size * GP_LOCK_NUMS
GP_USE_FREQ=0    # 是否使用频率文件进行图分区
GP_CUT=4096      # 图的度数将被限制在4096

##############
#   Search   #
#    搜索     #
##############
BM_LIST=(1)      # Beam宽度列表
T_LIST=(8)       # 线程数列表

CACHE=0          # 缓存节点数
MEM_L=0          # 内存搜索列表大小(非零启用)
# Page Search 页面搜索
USE_PAGE_SEARCH=1  # 搜索模式: 0=beam search, 1=page search(默认)
PS_USE_RATIO=1.0   # 页面搜索使用比例

# KNN K近邻搜索
LS="100"