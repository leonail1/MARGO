#!/bin/sh

# 通过调用所需的函数在config_local.sh文件中切换数据集

#################
#    SIFT100K   #
#################
dataset_SIFT100K() {
  BASE_PATH=../data/sift100k/sift_learn.fbin      # 基础数据集路径
  QUERY_FILE=../data/sift100k/sift_query.fbin     # 查询文件路径
  GT_FILE=../data/sift100k/sift_query_learn_gt100 # 真实结果文件路径
  PREFIX=sift100k                                  # 数据集前缀名称
  DATA_TYPE=float                                  # 数据类型
  DIST_FN=l2                                       # 距离函数(L2欧氏距离)
  B=0.003                                          # 构建参数B
  K=10                                             # K近邻的K值
  DATA_DIM=128                                     # 数据维度
  DATA_N=100000                                    # 数据点数量
}