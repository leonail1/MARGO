// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann {

  struct Neighbor {
    unsigned id;
    float    distance;
    bool     flag;
    unsigned rev_id = 0; // where is this neighbor comes from

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f)
        : id{id}, distance{distance}, flag(f) {
    }

    Neighbor(unsigned id, float distance, bool f, unsigned r_id)
        : id{id}, distance{distance}, flag(f), rev_id{r_id} {
    }

    inline bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
    inline bool operator==(const Neighbor &other) const {
      return (id == other.id);
    }
  };

  struct MemNavNeighbor {
    unsigned id;
    float distance;
    unsigned tag;

    MemNavNeighbor(unsigned i, float d, unsigned t)
        : id{i}, distance{d}, tag{t} {
    }
  };


  struct SimpleNeighbor {
    unsigned id;
    float    distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id, float distance) : id(id), distance(distance) {
    }

    inline bool operator<(const SimpleNeighbor &other) const {
      return distance < other.distance;
    }

    inline bool operator==(const SimpleNeighbor &other) const {
      return id == other.id;
    }
  };
  struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
  };

  /**
   * @brief 将新邻居插入到有序候选池中
   * 
   * 使用二分查找定位插入位置，然后通过memmove移动元素来保持数组有序。
   * 如果新节点的ID已存在于池中，则拒绝插入（返回K+1）。
   * 
   * @param addr 候选池数组指针
   * @param K 当前候选池大小
   * @param nn 待插入的新邻居
   * @return 插入位置索引；如果节点已存在则返回K+1
   */
  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K,
                                        Neighbor nn) {
    // 二分查找插入位置
    unsigned left = 0, right = K - 1;
    
    // 快速路径：新节点距离最小，插入到最前面
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    
    // 快速路径：新节点距离最大，插入到最后
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    
    // 二分查找定位插入区间 [left, right]
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;  // 插入位置在左半部分
      else
        left = mid;   // 插入位置在右半部分
    }

    // 去重检查：从left向左扫描距离相同的节点，检查ID是否重复
    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;  // 节点已存在，拒绝插入
      left--;
    }
    
    // 检查边界节点是否重复
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    
    // 在right位置插入，将right及其后续元素右移
    memmove((char *) &addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
  }

  // This class maintains a fixed-size sorted vector which supports
  //    1. if the vector is full and the distance of the inserting element is
  //       larger than the last element, early return. Otherwise, O(logN) insert
  //    2. move the first `num` elements to another vector
  class NeighborVec {
  public:
    void set_cap(size_t cap) {
      v.resize(cap+1);
      cap_ = cap;
      size_ = std::min(size_, cap_);
    }

    void insert(const Neighbor &nn) {
      if (size_ == 0 && cap_) {
        v[size_] = nn;
      } else {
        if (size_ == cap_ && nn.distance >= v[size_-1].distance) return;
        InsertIntoPool(v.data(), size_, nn);
      }

      if (size_ < cap_) ++size_;
    }

    size_t move_to(std::vector<Neighbor>& des, size_t des_idx, size_t num) {
      if (num > size_) {
        std::cout << "warning: require more neighbors than having. num: " << num << " size_: " << size_ << std::endl;
      }
      num = std::min(num, size_);
      if (des_idx + num >= des.size()) {
        std::cerr << "des size error" << std::endl;
        exit(1);
      }
      memmove(&(des.data()[des_idx]), v.data(), num * sizeof(Neighbor));
      if (size_ - num) {
        memmove(v.data(), &(v.data()[num]), (size_ - num)*sizeof(Neighbor));
      }
      size_ -= num;
      return num;
    }
  private:
    std::vector<Neighbor> v;
    size_t cap_ = 0, size_ = 0;
  };

  static inline unsigned InsertIntoPool(Neighbor *addr, unsigned K,
                                        Neighbor nn, NeighborVec& kicked, unsigned L) {
    // find the location to insert
    unsigned left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
      memmove((char *) &addr[left + 1], &addr[left], K * sizeof(Neighbor));
      if (K == L) kicked.insert(addr[K]);
      addr[left] = nn;
      return left;
    }
    if (addr[right].distance < nn.distance) {
      addr[K] = nn;
      return K;
    }
    while (right > 1 && left < right - 1) {
      unsigned mid = (left + right) / 2;
      if (addr[mid].distance > nn.distance)
        right = mid;
      else
        left = mid;
    }

    while (left > 0) {
      if (addr[left].distance < nn.distance)
        break;
      if (addr[left].id == nn.id)
        return K + 1;
      left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
      return K + 1;
    memmove((char *) &addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    if (K == L) kicked.insert(addr[K]);
    addr[right] = nn;
    return right;
  }
}  // namespace diskann