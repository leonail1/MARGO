#!/bin/bash

# 遇到错误立即退出
set -e
# set -x

# 加载本地配置文件
source config_local.sh

# 索引文件路径前缀
INDEX_PREFIX_PATH="../index/${PREFIX}_M${M}_R${R}_L${BUILD_L}_B${B}/"
# 内存采样路径
MEM_SAMPLE_PATH="${INDEX_PREFIX_PATH}SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
# 内存索引路径
MEM_INDEX_PATH="${INDEX_PREFIX_PATH}MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_MEM_USE_FREQ${MEM_USE_FREQ}_RANDOM_RATE${MEM_RAND_SAMPLING_RATE}_FREQ_RATE${MEM_FREQ_USE_RATE}/"
# 图分区路径
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_GP_USE_FREQ${GP_USE_FREQ}_CUT${GP_CUT}/"
# 频率文件路径
FREQ_PATH="${INDEX_PREFIX_PATH}FREQ/NQ_${FREQ_QUERY_CNT}_BM_${FREQ_BM}_L_${FREQ_L}_T_${FREQ_T}/"

# 汇总日志文件路径
SUMMARY_FILE_PATH="../indices/summary.log"

# 打印使用说明并退出
print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/freq/gp/search] [knn/range]"
  exit 1
}

# 检查目录是否存在,如果不存在则创建
check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

# 根据构建类型选择编译模式
case $1 in
  debug)
    # Debug模式编译
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ../debug
    EXE_PATH=../debug
  ;;
  release)
    # Release模式编译
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release
    EXE_PATH=../release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
# 切换到编译目录并执行编译
pushd $EXE_PATH
make -j
popd

# 创建索引目录并切换到该目录
mkdir -p ../indices && cd ../indices

# 打印当前日期
date
# 根据操作类型执行相应的任务
case $2 in
  build)
    # 构建磁盘索引
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/my_build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      -R $R \
      -L $BUILD_L \
      -B $B \
      -M $M \
      -T $BUILD_T > ${INDEX_PREFIX_PATH}build.log
    # 复制索引文件用于beam search
    cp ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index
  ;;
  sq)
    # 标量量化(Scalar Quantization)
    cp  ${INDEX_PREFIX_PATH}_disk_beam_search.index ${INDEX_PREFIX_PATH}_disk.index 
    time ${EXE_PATH}/tests/utils/sq ${INDEX_PREFIX_PATH} > ${INDEX_PREFIX_PATH}sq.log
  ;;
  build_mem)
    # 构建内存索引
    if [ ${MEM_USE_FREQ} -eq 1 ]; then
      # 如果使用频率文件
      if [ ! -d ${FREQ_PATH} ]; then
        echo "Seems you have not gen the freq file, run this script again: ./run_benchmark.sh [debug/release] freq [knn/range]"
        exit 1;
      fi
      echo "Parsing freq file..."
      time ${EXE_PATH}/tests/utils/parse_freq_file ${DATA_TYPE} ${BASE_PATH} ${FREQ_PATH}_freq.bin ${FREQ_PATH} ${MEM_FREQ_USE_RATE} 
      MEM_DATA_PATH=${FREQ_PATH}
    else
      # 使用随机采样
      mkdir -p ${MEM_SAMPLE_PATH}
      echo "Generating random slice..."
      time ${EXE_PATH}/tests/utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $MEM_RAND_SAMPLING_RATE > ${MEM_SAMPLE_PATH}sample.log
      MEM_DATA_PATH=${MEM_SAMPLE_PATH}
    fi
    echo "Building memory index..."
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_DATA_PATH} \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  freq)
    # 生成频率文件
    check_dir_and_make_if_absent ${FREQ_PATH}
    FREQ_LOG="${FREQ_PATH}freq.log"

    # 选择磁盘索引文件
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f $DISK_FILE_PATH ]; then
      DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    fi

    echo "Generating frequency file... ${FREQ_LOG}"
    time ${EXE_PATH}/tests/search_disk_index_save_freq \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --freq_save_path $FREQ_PATH \
              --query_file $FREQ_QUERY_FILE \
              --expected_query_num $FREQ_QUERY_CNT \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${FREQ_PATH}result \
              --num_nodes_to_cache ${FREQ_CACHE} \
              -T $FREQ_T \
              -L $FREQ_L \
              -W $FREQ_BM \
              --mem_L ${FREQ_MEM_L} \
              --use_page_search 0 \
              --disk_file_path ${DISK_FILE_PATH} > ${FREQ_LOG}
  ;;
  gp)
    # 图分区操作
    check_dir_and_make_if_absent ${GP_PATH}
    OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
    if [ ! -f "$OLD_INDEX_FILE" ]; then
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
    fi
    # 使用SQ索引文件进行图分区
    GP_DATA_TYPE=$DATA_TYPE
    if [ $USE_SQ -eq 1 ]; then 
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      GP_DATA_TYPE=uint8
    fi
    GP_FILE_PATH=${GP_PATH}_part.bin
    echo "Running graph partition... ${GP_FILE_PATH}.log"
    if [ ${GP_USE_FREQ} -eq 1 ]; then
      # 使用频率文件进行图分区
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES --freq_file ${FREQ_PATH}_freq.bin --lock_nums ${GP_LOCK_NUMS} --cut ${GP_CUT} > ${GP_FILE_PATH}.log
    else
      # 不使用频率文件进行图分区
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES > ${GP_FILE_PATH}.log
    fi

    echo "Running relayout... ${GP_PATH}relayout.log"
    time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} > ${GP_PATH}relayout.log
    if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
      mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
    fi
    # TODO: 只使用一个索引文件
    cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
    cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
  ;;
  search)
    # 搜索操作
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # 根据设置选择磁盘索引文件
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $USE_PAGE_SEARCH -eq 1 ]; then
      # 使用页面搜索
      if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
        echo "Partition file not found. Run the script with gp option first."
        exit 1
      fi
      echo "Using Page Search"
    else
      # 使用Beam搜索
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ -f ${OLD_INDEX_FILE} ]; then
        DISK_FILE_PATH=$OLD_INDEX_FILE
      else
        echo "make sure you have not gp the index file"
      fi
      echo "Using Beam Search"
    fi

    # 日志文件数组
    log_arr=()
    case $3 in
      knn)
        # K近邻搜索
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_SQ${USE_SQ}_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_MEM_USE_FREQ${MEM_USE_FREQ}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_USE_FREQ{$GP_USE_FREQ}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            # 清空缓存并执行搜索
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --use_sq ${USE_SQ}       > ${SEARCH_LOG} 
            log_arr+=( ${SEARCH_LOG} )
          done
        done
      ;;
      range)
        # 范围搜索
        for BW in ${BM_LIST[@]}
        do
          for T in ${T_LIST[@]}
          do
            SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_RADIUS${RADIUS}_CACHE${CACHE}_BW${BW}_T${T}_PS${USE_PAGE_SEARCH}_PS_RATIO${PS_USE_RATIO}_ITER_KNN${RS_ITER_KNN_TO_RANGE_SEARCH}_MEM_L${MEM_L}.log
            echo "Searching... log file: ${SEARCH_LOG}"
            # 清空缓存并执行范围搜索
            sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; ${EXE_PATH}/tests/range_search_disk_index \
              --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -W $BW \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              --range_threshold $RADIUS \
              -L $RS_LS \
              --disk_file_path ${DISK_FILE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --iter_knn_to_range_search ${RS_ITER_KNN_TO_RANGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_L ${MEM_L} \
              --custom_round_num ${RS_CUSTOM_ROUND} \
              --kicked_size ${KICKED_SIZE} \
              > ${SEARCH_LOG}
            log_arr+=( ${SEARCH_LOG} )
          done
        done
      ;;
      *)
        print_usage_and_exit
      ;;
    esac
    # 汇总所有搜索日志
    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat ${log_arr[0]} | grep -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH
        printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
        cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  *)
    print_usage_and_exit
  ;;
esac