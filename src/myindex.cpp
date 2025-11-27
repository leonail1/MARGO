
#include <iostream>
#include <omp.h>
#include <float.h>

#include "timer.h"
#include "myindex.h"


namespace diskann {

  template<typename T, typename TagT>
  myIndex<T, TagT>::myIndex(Metric m, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const bool enable_tags,
                        const bool support_eager_delete,
                        const bool concurrent_consolidate)
                         :Index<T, TagT>(m, dim, max_points, dynamic_index, enable_tags, 
                                         support_eager_delete, concurrent_consolidate) {
    const size_t total_internal_points = this->_max_points + this->_num_frozen_pts;
    this->weights_.resize(total_internal_points);
    this->wws_.resize(total_internal_points, 0.0f);
  }

  /**
   * @brief 从文件构建索引（MARGO图布局优化版本）
   * 
   * @param filename 数据文件路径
   * @param num_points_to_load 要加载的点数
   * @param parameters 构建参数（包含L、R、C、alpha等）
   * @param tags 标签向量（如果启用标签功能）
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::mybuild(const char *             filename,
                             const size_t             num_points_to_load,
                             Parameters &             parameters,
                             const std::vector<TagT> &tags) {
    std::cout << "running mybuild" << std::endl;
    // 检查点数是否为0
    if (num_points_to_load == 0)
      throw ANNException("Do not call build with 0 points", -1, __FUNCSIG__,
                         __FILE__, __LINE__);

    // 检查数据文件是否存在
    if (!file_exists(filename)) {
      diskann::cerr << "Data file " << filename
                    << " does not exist!!! Exiting...." << std::endl;
      std::stringstream stream;
      stream << "Data file " << filename << " does not exist." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      throw diskann::ANNException("Can not build with an empty file", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }

    // 读取文件元数据（点数和维度）
    diskann::get_bin_metadata(filename, file_num_points, file_dim);
    // 验证索引容量是否足够
    if (file_num_points > this->_max_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has " << file_num_points << " points, but "
             << "index can support only " << this->_max_points
             << " points as specified in constructor." << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // 验证请求加载的点数是否超过文件中的点数
    if (num_points_to_load > file_num_points) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << num_points_to_load
             << " points and file has only " << file_num_points << " points."
             << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // 验证数据维度是否匹配
    if (file_dim != this->_dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // 从文件中复制对齐后的数据
    copy_aligned_data_from_file<T>(filename, this->_data, file_num_points, file_dim,
                                   this->_aligned_dim);
    // 如果需要，归一化向量
    if (this->_normalize_vecs) {
      for (uint64_t i = 0; i < file_num_points; i++) {
        normalize(this->_data + this->_aligned_dim * i, this->_aligned_dim);
      }
    }

    diskann::cout << "Using only first " << num_points_to_load
                  << " from file.. " << std::endl;

    this->_nd = num_points_to_load;
    // 调用数据已填充的构建函数
    this->my_build_with_data_populated(parameters, tags);
  }

  /**
   * @brief 使用已加载的数据构建索引（MARGO核心构建函数）
   * 
   * 该函数实现MARGO论文中的图构建算法：
   * 1. 生成冻结点（frozen point）作为导航节点
   * 2. 调用mylink构建邻接图
   * 3. 计算边权重：基于单调可达性的度量
   *    - wws_[v]：能够单调到达顶点v的顶点个数（入度）
   *    - weights_[u][i]：边(u, neighbor[u][i])的权重 = wws_[u] × 基础权重
   * 4. 输出图统计信息（度数和权重分布）
   * 
   * @param parameters 构建参数（L、R、C、alpha等）
   * @param tags 标签向量（如果启用标签功能）
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_build_with_data_populated(Parameters &parameters, 
                                                      const std::vector<TagT> &tags) {
    std::cout << "running my build with data popoulated" << std::endl;
    diskann::cout << "Starting index build with " << this->_nd << " points... "
                  << std::endl;

    // 检查点数是否至少为1
    if (this->_nd < 1)
      throw ANNException("Error: Trying to build an index with 0 points", -1,
                         __FUNCSIG__, __FILE__, __LINE__);

    // 验证标签数量是否与点数匹配
    if (this->_enable_tags && tags.size() != this->_nd) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << this->_nd << " points from file,"
             << "but tags vector is of size " << tags.size() << "."
             << std::endl;
      diskann::cerr << stream.str() << std::endl;
      aligned_free(this->_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    // 如果启用标签，建立标签到位置的映射
    if (this->_enable_tags) {
      for (size_t i = 0; i < tags.size(); ++i) {
        this->_tag_to_location[tags[i]] = (unsigned) i;
        (this->_location_to_tag).set(static_cast<unsigned>(i), tags[i]);
      }
    }

    // 生成冻结点作为图的导航起点
	// 注意，当前代码下 MARGO 不支持在可更新图索引上做图布局优化，所以 generate_frozen_point 始终无效。
	// 查看 optimize_index_layout 获取更多信息
    this->generate_frozen_point();
    // 执行图链接过程（构建邻接图）
    this->mylink(parameters);

    // 如果支持即时删除，更新入边图
    if (this->_support_eager_delete) {
      this->update_in_graph();  // copying values to in_graph
    }

    // 步骤1: 计算每个节点的入度（能够单调到达该节点的顶点数）
    // wws_[nbr] 表示有多少个节点指向nbr
    #pragma omp parallel for schedule(dynamic, 4096)
    for( size_t i = 0; i < this->_nd; i++) {
      auto &pool = this->_final_graph[i];
      for( auto &nbr : pool) {
        LockGuard guard( this->_locks[nbr]);
        this->wws_[nbr] += 1.0f;  // nbr的入度加1
      }
    }

    // 步骤2: 计算边权重 = 起点入度 × 边的基础权重
    // 这样权重高的边表示：起点有更多前驱 → 该边能到达更多节点
    #pragma omp parallel for schedule(dynamic, 4096)
    for( size_t i = 0; i < this->_nd; i++) {
      auto &w_pool = this->weights_[i];
      for( auto &weight : w_pool) {
        weight *= this->wws_[i];  // 乘以起点i的入度
      }
    }

    // 统计并输出图的度数分布
    size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
    float max_w = 0.0f, min_w = FLT_MAX, total_w = 0.0f;
    size_t cnt_0 = 0;
    for (size_t i = 0; i < this->_nd; i++) {
      auto &pool = this->_final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size(); 
      if (pool.size() < 2)
        cnt++;
      auto &pool_w = this->weights_[i];
      max_w = (std::max)(max_w, *std::max_element(pool_w.begin(), pool_w.end()));
      min_w = (std::min)(min_w, *std::min_element(pool_w.begin(), pool_w.end()));
      total_w += std::accumulate(pool_w.begin(), pool_w.end(), 0);
      for( auto &tmp : pool_w) {
        if( tmp < 2) 
          cnt_0++;
      }
    }
    diskann::cout << "Index built with degree: max:" << max
                  << "  avg:" << (float) total / (float) (this->_nd + this->_num_frozen_pts)
                  << "  min:" << min << "  count(deg<2):" << cnt << std::endl;
    diskann::cout << "Index built with weight: max:" << max_w
                  << "  avg:" << (float) total_w / (float) (total)
                  << "  min:" << min_w 
                  << "  count(weight<2):" << cnt_0 << " #edges:" << total
                  << std::endl;

    this->_max_observed_degree = (std::max)((unsigned) max, this->_max_observed_degree);
    this->_has_built = true;
  }

  /**
   * @brief 构建图的链接结构（MARGO的邻接图构建算法）
   * 
   * 该函数实现了迭代式图构建过程：
   * 1. 初始化：设置参数（L、R、C、alpha）、确定起点、预留空间
   * 2. 多轮迭代（默认2轮）：
   *    - 每轮分批处理所有节点
   *    - 对每个节点：搜索候选邻居 → 剪枝 → 添加边 → 反向插入
   *    - 需要时重新剪枝（当度数超过阈值）
   * 3. 剪枝过程计算边权重（基于RNG准则和单调可达性）
   * 
   * @param parameters 构建参数
   *        - L: 搜索列表大小（候选邻居数）
   *        - R: 每个节点的最大出度
   *        - C: 剪枝时的最大候选数
   *        - alpha: RNG剪枝的alpha参数
   *        - num_threads: 并行线程数
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::mylink(Parameters &parameters) {
    std::cout << "running mylink" << std::endl;
    // 设置线程数
    unsigned num_threads = parameters.Get<unsigned>("num_threads");
    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    // 计算同步次数（将所有节点分批处理）
    uint32_t num_syncs =
        (unsigned) DIV_ROUND_UP(this->_nd + this->_num_frozen_pts, (64 * 64));
    if (num_syncs < 40)
      num_syncs = 40;
    diskann::cout << "Number of syncs: " << num_syncs << std::endl;

    this->_saturate_graph = parameters.Get<bool>("saturate_graph");

    if (num_threads != 0)
      omp_set_num_threads(num_threads);

    // 从参数中读取构建配置
    this->_indexingQueueSize = parameters.Get<unsigned>("L");  // 搜索列表大小
    this->_indexingRange = parameters.Get<unsigned>("R");      // 最大出度
    this->_indexingMaxC = parameters.Get<unsigned>("C");       // 剪枝候选数
    const float last_round_alpha = parameters.Get<float>("alpha");
    unsigned    L = this->_indexingQueueSize;

    // 设置多轮迭代（2轮）
    std::vector<unsigned> Lvec;
    Lvec.push_back(L);
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;
    this->_indexingAlpha = 1.0f;

    /**
     * visit_order: 节点访问顺序列表，包含所有数据点和冻结点的索引
     *              用于控制图构建过程中处理节点的顺序，初始化为所有节点ID [0, _nd-1] + 冻结点
     * pool: 候选邻居池，存储搜索过程中找到的候选邻居节点及其距离
     *       在贪心搜索阶段用于收集L个最近邻候选
     * tmp: 临时邻居列表，用于剪枝算法中的中间计算
     * visited: 已访问节点集合（哈希集），用于避免在搜索过程中重复访问同一节点
     */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(this->_nd + this->_num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) this->_nd; i++) {
      visit_order.emplace_back(i);
    }

    // 添加冻结点
    if (this->_num_frozen_pts > 0)
      visit_order.emplace_back((unsigned) this->_max_points);

    // 确定图的起点：优先使用冻结点，否则计算入口点
    if (this->_num_frozen_pts > 0)
      this->_start = (unsigned) this->_max_points;
    else
      this->_start = this->calculate_entry_point();

    // 如果支持即时删除，初始化入边图
    if (this->_support_eager_delete) {
      (this->_in_graph).reserve(this->_max_points + this->_num_frozen_pts);
      (this->_in_graph).resize(this->_max_points + this->_num_frozen_pts);
    }

    // 为每个节点的邻接列表预留空间（考虑松弛因子）
    for (uint64_t p = 0; p < this->_nd; p++) {
      this->_final_graph[p].reserve(
          (size_t)(std::ceil(this->_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
      this->weights_.reserve(
          (size_t)(std::ceil(this->_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
    }

    /**
     * rd: 随机设备，用于获取真随机数种子（基于硬件熵源）
     * gen: Mersenne Twister 19937 伪随机数生成器，使用rd()作为种子初始化
     *      MT19937是高质量的伪随机数生成器，周期长、分布均匀
     * dis: 均匀分布器，生成 [0, 1) 区间内的均匀分布浮点随机数
     *      注：虽然这里定义了随机数生成器，但在当前MARGO实现中未被使用
     */
    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // 创建初始搜索点列表（包含起点）
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(this->_start);

    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    // 开始多轮迭代构建
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      // 每轮开始时重置权重
      const size_t total_internal_points = this->_max_points + this->_num_frozen_pts;
      this->wws_.clear();
      this->wws_.resize( total_internal_points, 0.0f);
      #pragma omp parallel for schedule(dynamic, 4096)
      for( size_t i = 0; i < total_internal_points; i++) {
        for( auto &weight : this->weights_[i]) {
          weight = 1.0f;  // 重置为基础权重
        }
      }

      L = Lvec[rnd_no];

      // 最后一轮使用指定的alpha值
      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          this->_indexingAlpha = last_round_alpha;
      }

      /**
       * sync_time: 当前批次的同步时间（搜索+剪枝阶段的耗时）
       * total_sync_time: 累计的所有批次同步时间
       * inter_time: 当前批次的反向插入和重剪枝时间
       * total_inter_time: 累计的反向插入和重剪枝时间
       * inter_count: 当前批次需要重剪枝的节点数量
       * total_inter_count: 累计需要重剪枝的节点总数
       * progress_counter: 进度计数器，用于控制进度输出（避免输出过于频繁）
       */
      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      /**
       * round_size: 每批处理的节点数量，将所有节点均分成 num_syncs 批
       * need_to_sync: 标记数组，记录哪些节点因反向插入导致度数超限，需要重新剪枝
       *               索引为节点ID，值为0表示不需要同步，非0表示需要
       */
      size_t round_size = DIV_ROUND_UP(this->_nd, num_syncs);  // 每批的大小
      std::vector<unsigned> need_to_sync(this->_max_points + this->_num_frozen_pts, 0);

      /**
       * pruned_list_vector: 二维数组，存储每个节点剪枝后的邻居列表
       *                     第一维大小为 round_size（当前批次的节点数）
       *                     pruned_list_vector[i] 存储批次中第 i 个节点的剪枝邻居ID列表
       * weight_list_vector: 二维数组，存储每个节点剪枝后的边权重列表
       *                     与 pruned_list_vector 一一对应
       *                     weight_list_vector[i][j] 是第 i 个节点到其第 j 个邻居的边权重
       */
      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);
      std::vector<std::vector<float>> weight_list_vector(round_size);

      // 分批处理所有节点
      for (uint32_t sync_num = 0; sync_num < num_syncs; sync_num++) {
        size_t start_id = sync_num * round_size;
        size_t end_id =
            (std::min)(this->_nd + this->_num_frozen_pts, (sync_num + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;

        // 步骤1: 并行搜索和剪枝
        // 对每个节点搜索L个候选邻居，然后用RNG准则剪枝到R个
#pragma omp parallel for schedule(dynamic)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          auto                     node = visit_order[node_ctr];
          size_t                   node_offset = node_ctr - start_id;
          tsl::robin_set<unsigned> visited;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          std::vector<float> &weight_list = weight_list_vector[node_offset];
          // 搜索候选邻居（贪心搜索L个最近邻）
          std::vector<Neighbor> pool;
          pool.reserve(L * 2);
          visited.reserve(L * 2);
          this->get_expanded_nodes(node, L, init_ids, pool, visited);
          // 将当前邻居中未访问的节点加入候选池
          if (!this->_final_graph[node].empty())
            for (auto id : this->_final_graph[node]) {
              if (visited.find(id) == visited.end() && id != node) {
                float dist =
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                       this->_data + this->_aligned_dim * (size_t) id,
                                       (unsigned) this->_aligned_dim);
                pool.emplace_back(Neighbor(id, dist, true));
                visited.insert(id);
              }
            }
          // 使用RNG准则剪枝，并计算边权重
          this->my_prune_neighbors(node, pool, pruned_list, weight_list);
        }
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

        // 步骤2: 将剪枝后的邻居添加到图中
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          _u64                   node = visit_order[node_ctr];
          size_t                 node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          // 更新节点的邻接列表
          this->_final_graph[node].clear();
          for (auto id : pruned_list)
            this->_final_graph[node].emplace_back(id);
          // 更新边权重
          std::vector<float> &weight_list = weight_list_vector[node_offset];
          this->weights_[node].clear();
          for (auto w : weight_list)
            this->weights_[node].emplace_back(w);
          weight_list.clear();
          weight_list.shrink_to_fit();
        }

        s = std::chrono::high_resolution_clock::now();

        // 步骤3: 反向插入（将边 u->v 的反向边 v->u 也添加到图中）
#pragma omp parallel for schedule(dynamic, 64)
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id; ++node_ctr) {
          auto                   node = visit_order[node_ctr];
          _u64                   node_offset = node_ctr - start_id;
          std::vector<unsigned> &pruned_list = pruned_list_vector[node_offset];
          this->my_batch_inter_insert(node, pruned_list, need_to_sync);
          pruned_list.clear();
          pruned_list.shrink_to_fit();
        }

        // 步骤4: 对度数超过阈值的节点重新剪枝
#pragma omp parallel for schedule(dynamic, 65536)
        for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size());
             node_ctr++) {
          auto node = visit_order[node_ctr];
          if (need_to_sync[node] != 0) {
            need_to_sync[node] = 0;
            inter_count++;
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;
            std::vector<float>    new_out_weights;

            // 重新计算所有邻居的距离
            for (auto cur_nbr : this->_final_graph[node]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node) {
                float dist =
                    this->_distance->compare(this->_data + this->_aligned_dim * (size_t) node,
                                             this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                             (unsigned) this->_aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            // 重新剪枝
            this->my_prune_neighbors(node, dummy_pool, new_out_neighbors, new_out_weights);

            // 更新邻接列表和权重
            this->_final_graph[node].clear();
            for (auto id : new_out_neighbors)
              this->_final_graph[node].emplace_back(id);
            
            this->weights_[node].clear();
            for (auto w : new_out_weights)
              this->weights_[node].emplace_back(w);
          }
        }

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / num_syncs > progress_counter) {
          diskann::cout.precision(4);
          diskann::cout << "Completed  (round: " << rnd_no
                        << ", sync: " << sync_num << "/" << num_syncs
                        << " with L " << L << ")"
                        << " sync_time: " << sync_time << "s"
                        << "; inter_time: " << inter_time << "s" << std::endl;

          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 5;
        }
      }
// Gopal. Splitting nsg_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (this->_nd > 0) {
        diskann::cout << "Completed Pass " << rnd_no << " of data using L=" << L
                      << " and alpha=" << parameters.Get<float>("alpha")
                      << ". Stats: ";
        diskann::cout << "search+prune_time=" << total_sync_time
                      << "s, inter_time=" << total_inter_time
                      << "s, inter_count=" << total_inter_count << std::endl;
      }
    }

    if (this->_nd > 0) {
      diskann::cout << "Starting final cleanup.." << std::flush;
    }
#pragma omp parallel for schedule(dynamic, 65536)
    for (_s64 node_ctr = 0; node_ctr < (_s64)(visit_order.size()); node_ctr++) {
      auto node = visit_order[node_ctr];
      if (this->_final_graph[node].size() > this->_indexingRange) {
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);
        std::vector<unsigned>    new_out_neighbors;
        std::vector<float>    new_out_weights;

        for (auto cur_nbr : this->_final_graph[node]) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != node) {
            float dist =
                (this->_distance)->compare(this->_data + this->_aligned_dim * (size_t) node,
                                           this->_data + this->_aligned_dim * (size_t) cur_nbr,
                                           (unsigned) this->_aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        this->my_prune_neighbors(node, dummy_pool, new_out_neighbors, new_out_weights);

        this->_final_graph[node].clear();
        for (auto id : new_out_neighbors)
          this->_final_graph[node].emplace_back(id);
        
        this->weights_[node].clear();
          for (auto w : new_out_weights)
            this->weights_[node].emplace_back(w);
      }
    }
    if (this->_nd > 0) {
      diskann::cout << "done. Link time: "
                    << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                    << std::endl;
    }
  }

  /**
   * @brief 邻居剪枝函数（简化版本，使用默认参数）
   * 
   * @param location 当前节点ID
   * @param pool 候选邻居池（已按距离排序）
   * @param pruned_list 输出：剪枝后的邻居ID列表
   * @param weight_list 输出：对应的边权重列表
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_prune_neighbors(const unsigned        location,
                                            std::vector<Neighbor> &pool,
                                            std::vector<unsigned> &pruned_list,
                                            std::vector<float> &weight_list) {
    my_prune_neighbors(location, pool, this->_indexingRange, this->_indexingMaxC,
                       this->_indexingAlpha, pruned_list, weight_list);
  }

  /**
   * @brief 使用RNG准则对邻居进行剪枝并计算边权重（MARGO核心算法）
   * 
   * 该函数实现了带权重的Robust Neighbor Graph (RNG)剪枝：
   * 1. 对候选池按距离排序
   * 2. 使用my_occlude_list应用RNG准则，计算每条边的权重
   * 3. 权重计算基于：该边在剪枝过程中"遮挡"了多少个候选点
   * 4. 如果启用saturate_graph，用最近的点填充到range个邻居
   * 
   * 边权重的意义：
   * - 如果边(location, p)遮挡了很多候选点，说明它的"覆盖能力"强
   * - 权重高的边更可能被用于到达更多的其他节点
   * 
   * @param location 当前节点ID
   * @param pool 候选邻居池（会被排序）
   * @param range 最大邻居数（R参数）
   * @param max_candidate_size 最大候选数（C参数）
   * @param alpha RNG剪枝的alpha参数（控制遮挡程度）
   * @param pruned_list 输出：剪枝后的邻居ID列表
   * @param weight_list 输出：对应的边权重列表
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_prune_neighbors(const unsigned        location,
                                            std::vector<Neighbor> &pool,
                                            const _u32            range,
                                            const _u32  max_candidate_size,
                                            const float alpha,
                                            std::vector<unsigned> &pruned_list,
                                            std::vector<float> &weight_list) {
    if (pool.size() == 0) {
      std::stringstream ss;
      ss << "Thread loc:" << std::this_thread::get_id()
         << " Pool address: " << &pool << std::endl;
      std::cout << ss.str();
      throw diskann::ANNException("Pool passed to prune_neighbors is empty",
                                  -1);
    }

    this->_max_observed_degree = (std::max)(this->_max_observed_degree, range);

    // 按距离排序候选池（从近到远）
    std::sort(pool.begin(), pool.end());

    // 存储剪枝结果：每个邻居及其对应的权重
    std::vector<std::pair<Neighbor, float>> result;
    result.reserve(range);

    // 应用RNG准则进行剪枝，并计算边权重
    this->my_occlude_list(pool, alpha, range, max_candidate_size, result, location);

    // 提取邻居ID和权重到输出列表
    pruned_list.clear();
    weight_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if( iter.first.id != location) {
        pruned_list.emplace_back(iter.first.id);
        weight_list.emplace_back(iter.second);
      }
    }

    // 如果启用饱和图且alpha>1，用最近的点填充到range个邻居
    if (this->_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
             pruned_list.end()) &&
            pool[i].id != location) {
          pruned_list.emplace_back(pool[i].id);
          weight_list.emplace_back(1.0f);  // 填充的边权重为1.0
        }
      }
    }
  }

  /**
   * @brief RNG遮挡列表剪枝算法，计算边权重（MARGO权重计算核心）
   * 
   * 该函数实现了Robust Neighbor Graph (RNG)的遮挡准则，并在剪枝过程中计算边权重。
   * 使用论文中的记号：p（当前节点），p*（候选邻居），p'（被遮挡的点）
   * 
   * 算法流程：
   * 1. 从距离最近的候选点开始遍历（已排序）
   * 2. 对每个候选点p*，检查它是否遮挡后续的候选点p'
   * 3. 遮挡判断（L2/COSINE距离）：
   *    - 如果 dist(p*, p') == 0：完全遮挡
   *    - 如果 dist(p, p') / dist(p*, p') > alpha：p' 被 p* 遮挡
   * 4. 权重计算：边(p, p*)的权重 = 1.0 + 它遮挡的边的数量
   * 5. 同时累加 wws_[p']，用于后续计算入度
   * 
   * 理论基础（来自MARGO论文）：
   * 
   * Lemma 3: 在SNG中，对于任意边(p, p*)，它能够单调到达的顶点数量
   *          等于它遮挡的边的数量加1。
   * 
   * 证明要点：
   * - 必要性：如果路径[p, p*, ..., p']是单调路径，则有 dist(p*, p') < dist(p, p')
   *          根据SNG的短边优先规则，(p, p*)遮挡(p, p')
   * - 充分性：如果(p, p*)遮挡(p, p')，则 dist(p, p') > dist(p*, p')
   *          SNG的性质保证存在从p*到p'的单调路径
   * 
   * 权重的物理意义：
   * - 边(p, p*)的权重 = 1 + 遮挡的边数 = 它能单调到达的顶点数量
   * - 这是对"单调可达性"的精确计算（基于RNG遮挡关系）
   * - 权重越高，该边的"覆盖能力"越强，在图布局中应优先保持在同一页面
   * 
   * 变量映射：
   * - location → p（当前查询节点）
   * - iter → p*（候选邻居，将被选入结果）
   * - iter2 → p'（被p*遮挡的候选点）
   * 
   * 多轮alpha策略：
   * - 从alpha=1.0开始，逐步放松到指定的alpha值（每轮乘以1.2）
   * - 确保即使在严格的RNG准则下也能选出足够的邻居
   * 
   * @param pool 候选邻居池（已排序，会被截断到maxc个）
   * @param alpha RNG剪枝参数（控制遮挡的严格程度）
   * @param degree 目标邻居数量（最多选这么多个）
   * @param maxc 最大候选数（超过此数量会被截断）
   * @param result 输出：选中的邻居及其权重的pair列表
   * @param location 当前节点ID（对应论文中的p）
   */
  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_occlude_list(std::vector<Neighbor> &pool,
                                         const float alpha, const unsigned degree,
                                         const unsigned         maxc,
                                         std::vector<std::pair<Neighbor, float>> &result, 
                                         const unsigned location) {
    if (pool.size() == 0)
      return;

    assert(std::is_sorted(pool.begin(), pool.end()));
    // 截断候选池到maxc个
    if (pool.size() > maxc)
      pool.resize(maxc);
    // 记录每个候选点的遮挡因子（被遮挡的程度）
    std::vector<float> occlude_factor(pool.size(), 0);

    // 从alpha=1.0开始，逐步放松到指定alpha
    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      // 用于MIPS距离的特殊标记（标记已被剪枝的点）
      float eps = cur_alpha + 0.01f;

      // 遍历候选池，选择未被遮挡的点作为p*
      for (auto iter = pool.begin();
           result.size() < degree && iter != pool.end(); ++iter) {
        // 如果该点已被遮挡，跳过
        if (occlude_factor[iter - pool.begin()] > cur_alpha) {
          continue;
        }
        // 标记该点p*已被选中（不会再被其他点遮挡）
        occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
        
        // 初始化边(p, p*)的权重：基础权重1.0
        float weight = 1.0f;
        // 如果边(p, p*)已经存在于图中，继承之前的权重
		// location 对应 p，iter 对应 p*
        auto tmp_iter = std::find(this->_final_graph[location].begin(), 
                                  this->_final_graph[location].end(), iter->id);
        if( tmp_iter != this->_final_graph[location].end()) {
          weight = this->weights_[location][tmp_iter - this->_final_graph[location].begin()];
        }
        
        // 检查p*是否遮挡后续的候选点p'
        // 根据Lemma 3：遮挡的边数 = 单调可达的顶点数 - 1
        for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
          auto t = iter2 - pool.begin();
          // 如果p'已被严重遮挡，跳过
          if (occlude_factor[t] > alpha)
            continue;
          
          // 计算 dist(p*, p')
		  // iter2 对应 p'
          float djk =
              (this->_distance)->compare(this->_data + this->_aligned_dim * (size_t) iter2->id,
                                         this->_data + this->_aligned_dim * (size_t) iter->id,
                                         (unsigned) this->_aligned_dim);
          
          // L2或余弦距离的遮挡判断
          if (this->_dist_metric == diskann::Metric::L2 ||
              this->_dist_metric == diskann::Metric::COSINE) {
            // 情况1：p*和p'距离为0（完全重合）
            if (djk == 0.0) {
              occlude_factor[t] = std::numeric_limits<float>::max();
              // (p, p*)遮挡了边(p, p')，权重加1
              weight += 1.0f;
              {
                LockGuard guard( this->_locks[iter2->id]);
                this->wws_[iter2->id] += 1.0f;  // 累加p'的入度候选计数
              }
            }
            // 情况2：检查RNG遮挡条件
            else {
              // ratio_dist = dist(p, p') / dist(p*, p')
              float ratio_dist = iter2->distance / djk;
              // 如果 ratio_dist > cur_alpha，说明边(p, p')被边(p, p*)遮挡
              // 根据短边优先规则：dist(p, p') > dist(p*, p') 意味着存在单调路径
              if( occlude_factor[t] <= cur_alpha && ratio_dist > cur_alpha) {
                // (p, p*)遮挡了边(p, p')，权重加1
                weight += 1.0f;
                {
                  LockGuard guard( this->_locks[iter2->id]);
                  this->wws_[iter2->id] += 1.0f;  // 累加p'的入度候选计数
                }
              }
              // 更新p'的遮挡因子
              occlude_factor[t] = std::max(occlude_factor[t], ratio_dist);
            }
          } 
          // 内积距离的特殊处理（MIPS）
          else if (this->_dist_metric == diskann::Metric::INNER_PRODUCT) {
            // 翻转距离（因为MIPS中距离越大越好）
            float x = -iter2->distance;
            float y = -djk;
            if (y > cur_alpha * x) {
              if( occlude_factor[t] < eps - 1e-6) {
                // (p, p*)遮挡了边(p, p')，权重加1
                weight += 1.0f;
                {
                  LockGuard guard( this->_locks[iter2->id]);
                  this->wws_[iter2->id] += 1.0f;  // 累加p'的入度候选计数
                }
              }
              occlude_factor[t] = std::max(occlude_factor[t], eps);
            }
          }
        }
        // 将选中的邻居p*及其计算的权重加入结果
        // 此时 weight = 1 + 遮挡的边数 = 边(p, p*)能单调到达的顶点数
        result.emplace_back(std::make_pair( *iter, weight));
      }
      // 放松alpha值，进入下一轮
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_batch_inter_insert(
          unsigned n, const std::vector<unsigned> &pruned_list, const _u32 range,
          std::vector<unsigned> &need_to_sync) {
    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      // des.loc is the loc of the neighbors of n
      assert(des >= 0 && des < this->_max_points + this->_num_frozen_pts);
      if (des > this->_max_points)
        diskann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      // des_pool contains the neighbors of the neighbors of n

      {
        LockGuard guard(this->_locks[des]);
        if (std::find(this->_final_graph[des].begin(), this->_final_graph[des].end(), n) ==
            this->_final_graph[des].end()) {
          this->_final_graph[des].push_back(n);
          this->weights_[des].push_back(1.0f);
          // this->wws_[des] += 1.0f;
          if (this->_final_graph[des].size() >
              (unsigned) (range * GRAPH_SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::my_batch_inter_insert(
          unsigned n, const std::vector<unsigned> &pruned_list,
          std::vector<unsigned> &need_to_sync) {
    my_batch_inter_insert(n, pruned_list, this->_indexingRange, need_to_sync);
  }

  template<typename T, typename TagT>
  void myIndex<T, TagT>::mysave(const char *filename, bool compact_before_save) {
    std::cout << "running mysave" << std::endl;
    diskann::Timer timer;

    if (compact_before_save) {
      this->compact_data();
      this->compact_frozen_point();
    } else {
      if (not this->_data_compacted) {
        throw ANNException(
            "Index save for non-compacted index is not yet implemented", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    std::unique_lock<std::shared_timed_mutex> ul(this->_update_lock);
    std::unique_lock<std::shared_timed_mutex> cl(this->_consolidate_lock);
    std::unique_lock<std::shared_timed_mutex> tl(this->_tag_lock);
    std::unique_lock<std::shared_timed_mutex> dl(this->_delete_lock);

    if (!this->_save_as_one_file) {
      std::string graph_file = std::string(filename);
      std::string tags_file = std::string(filename) + ".tags";
      std::string data_file = std::string(filename) + ".data";
      std::string delete_list_file = std::string(filename) + ".del";
      std::string weights_file = std::string(filename) + ".weights";

      // Because the save_* functions use append mode, ensure that
      // the files are deleted before save. Ideally, we should check
      // the error code for delete_file, but will ignore now because
      // delete should succeed if save will succeed.
      delete_file(graph_file);
      this->save_graph(graph_file);
      delete_file(data_file);
      this->save_data(data_file);
      delete_file(tags_file);
      this->save_tags(tags_file);
      delete_file(delete_list_file);
      this->save_delete_list(delete_list_file);
      delete_file(weights_file);
      this->my_save_weights(weights_file);
    } else {
      diskann::cout << "Save index in a single file currently not supported. "
                       "Not saving the index."
                    << std::endl;
    }

    this->reposition_frozen_point_to_end();

    diskann::cout << "Time taken for save: " << timer.elapsed() / 1000000.0
                  << "s." << std::endl;
  }

  template<typename T, typename TagT>
  _u64 myIndex<T, TagT>::my_save_weights(std::string graph_file) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    _u64 file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);

    float max_weight = 0.0f;
    out.write((char *) &max_weight, sizeof(float));
    _u64 num_edges = 0;
    out.write((char *) &num_edges, sizeof(_u64));
    out.write((char *) &this->_nd, sizeof(_u64));
    for (unsigned i = 0; i < this->_nd; i++) {
      unsigned out_degree = (unsigned) this->weights_[i].size();
      out.write((char *) &out_degree, sizeof(unsigned));
      out.write((char *) this->weights_[i].data(), out_degree * sizeof(float));
      max_weight = std::max( max_weight,
                             *std::max_element( this->weights_[i].begin(), 
                                                this->weights_[i].end()));
      num_edges += out_degree;
    }
    out.seekp(file_offset, out.beg);
    out.write((char *) &max_weight, sizeof(float));
    out.write((char *) &num_edges, sizeof(_u64));
    out.close();
    std::cout << "saving weights..."
              << ", #edges:" << num_edges
              << ", #vertice:" << this->_nd
              << ", max weight:" << max_weight
              << std::endl;
    return num_edges;  // number of bytes written
  }

  // EXPORTS
  template DISKANN_DLLEXPORT class myIndex<float, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, int32_t>;
  template DISKANN_DLLEXPORT class myIndex<float, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, uint32_t>;
  template DISKANN_DLLEXPORT class myIndex<float, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, int64_t>;
  template DISKANN_DLLEXPORT class myIndex<float, uint64_t>;
  template DISKANN_DLLEXPORT class myIndex<int8_t, uint64_t>;
  template DISKANN_DLLEXPORT class myIndex<uint8_t, uint64_t>;
}