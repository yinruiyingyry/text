// Copyright 2022 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "absl/types/span.h"
#include "tensorflow_text/core/kernels/trimmer.h"


namespace tensorflow {
namespace text {

template <typename T, typename Tsplits = int32_t>
class RoundRobinTrimmer : Trimmer<T, Tsplits> {
  using Mask = std::vector<bool>;
  using Segment = std::vector<T>;
  using SegmentSpan = absl::Span<T>;
  using RowSplits = std::vector<Tsplits>;
  using RowSplitsSpan = absl::Span<Tsplits>;

 public:
  RoundRobinTrimmer(int max_sequence_length)
      : max_sequence_length_(max_sequence_length) {}
  virtual ~RoundRobinTrimmer() = default;

  // Generates masks for a single batch of segments.
  std::vector<Mask> GenerateMasks(
      const std::vector<Segment>& segments) const;

  // Generates masks for a batch of segment row splits.
  //
  // The returned value is a flattened list of mask values which can be split
  // into batches using the same input row splits.
  std::vector<Mask> GenerateMaskBatches(
      const std::vector<RowSplits>& row_splits) const;
  std::vector<Mask> GenerateMaskBatches(
      const std::vector<RowSplitsSpan>& row_splits) const;

  // Trims a single batch of segments.
  void Trim(std::vector<Segment>* segments) const;

  // Trims a batch of segments given their flattened values and row splits.
  //
  // The returned values are the flattened trimmed values and new row splits.
  std::pair<std::vector<Segment>, std::vector<RowSplits>> Trim(
      const std::vector<Segment>& flat_segments,
      const std::vector<RowSplits>& row_splits) const;
  std::pair<std::vector<Segment>, std::vector<RowSplits>> Trim(
      const std::vector<SegmentSpan>& flat_segments,
      const std::vector<RowSplitsSpan>& row_splits) const;

 private:
  // Used for holding data about segment sizes and how much of it is used.
  struct Row {
    Row() : used(0) {}
    Row(int idx, int size, int used) : idx(idx), size(size), used(used) {}
    int idx;       // Index into the list of segments
    Tsplits size;  // Size of the row segment
    int used;      // How much of the segment is used
  };

  // Internal execution to share code for Span & Vector row_splits.
  template <typename Container>
  std::vector<Mask> GenerateMasksInternal(
      const std::vector<Container>& row_splits) const;

  // Internal execution to share code for Span & Vector row_splits.
  template <typename ContainerVals, typename ContainerSplits>
  std::pair<std::vector<Segment>, std::vector<RowSplits>> TrimInternal(
      const std::vector<ContainerVals>& flat_segments,
      const std::vector<ContainerSplits>& row_splits) const;

  // Main process of the timmer. Process row splits a batch at a time. Once each
  // it is known how much each row in a batch is used, the callback is called
  // with the row information.
  // Algorithm to fill segments:
  // 1. Fill segments that will max starting from smallest to largest.
  // 2. Partially fill the rest up the same amount up to the sequence length.
  // 3. Add the remainder to the available rows in order.
  void ProcessSplitsByBatch(const std::vector<RowSplits>& row_splits,
      std::function<void(std::vector<Row>)> callback) const;

  void ProcessBatch(std::vector<Row>* segment_sizes) const;
  void ProcessSplitsByBatch(const std::vector<RowSplitsSpan>& row_splits,
      std::function<void(std::vector<Row>)> callback) const;

  const int max_sequence_length_;
};

/******************************* Implementation *******************************/

template <typename T, typename Tsplits>
std::vector<std::vector<bool>>
RoundRobinTrimmer<T, Tsplits>::GenerateMasks(
    const std::vector<Segment>& segments) const {
  int num_segments = segments.size();
  // Get size of each segment
  std::vector<Row> segment_sizes(num_segments);
  for (int s = 0; s < num_segments; ++s) {
    segment_sizes[s].idx = s;
    segment_sizes[s].size = segments[s].size();
  }
  ProcessBatch(&segment_sizes);
  std::vector<Mask> masks(num_segments);
  for (int i = 0; i < num_segments; ++i) {
    Mask& m = masks[i];
    m.reserve(segment_sizes[i].size);
    m.insert(m.end(), segment_sizes[i].used, true);
    m.insert(m.end(), segment_sizes[i].size - segment_sizes[i].used, false);
  }
  return masks;
}

template <typename T, typename Tsplits>
std::vector<std::vector<bool>>
RoundRobinTrimmer<T, Tsplits>::GenerateMaskBatches(
    const std::vector<RowSplits>& row_splits) const {
  return GenerateMasksInternal<std::vector<Tsplits>>(row_splits);
}

template <typename T, typename Tsplits>
std::vector<std::vector<bool>>
RoundRobinTrimmer<T, Tsplits>::GenerateMaskBatches(
    const std::vector<RowSplitsSpan>& row_splits) const {
  return GenerateMasksInternal<absl::Span<Tsplits>>(row_splits);
}

template <typename T, typename Tsplits>
template <typename Container>
std::vector<std::vector<bool>>
RoundRobinTrimmer<T, Tsplits>::GenerateMasksInternal(
    const std::vector<Container>& row_splits) const {
  std::vector<Mask> masks(row_splits.size());
  for (int i = 0; i < row_splits.size(); ++i) {
    masks[i].reserve(row_splits[i].back());
  }
  ProcessSplitsByBatch(row_splits, [&masks](std::vector<Row> rows) {
    for (int s = 0; s < masks.size(); ++s) {
      masks[s].insert(masks[s].end(), rows[s].used, true);
      masks[s].insert(masks[s].end(), rows[s].size - rows[s].used, false);
    }
  });
  return masks;
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::Trim(
    std::vector<Segment>* segments) const {
  int num_segments = segments->size();
  std::vector<Row> segment_sizes(num_segments);
  for (int s = 0; s < num_segments; ++s) {
    segment_sizes[s].idx = s;
    segment_sizes[s].size = (*segments)[s].size();
  }
  ProcessBatch(&segment_sizes);
  for (int s = 0; s < num_segments; ++s) {
    (*segments)[s].resize(segment_sizes[s].used);
  }
}

template <typename T, typename Tsplits>
std::pair<std::vector<std::vector<T>>, std::vector<std::vector<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::Trim(
    const std::vector<Segment>& flat_segments,
    const std::vector<RowSplits>& row_splits) const {
  return TrimInternal<std::vector<T>, std::vector<Tsplits>>(flat_segments,
                                                            row_splits);
}

template <typename T, typename Tsplits>
std::pair<std::vector<std::vector<T>>, std::vector<std::vector<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::Trim(
    const std::vector<SegmentSpan>& flat_segments,
    const std::vector<RowSplitsSpan>& row_splits) const {
  return TrimInternal<absl::Span<T>, absl::Span<Tsplits>>(flat_segments,
                                                          row_splits);
}

template <typename T, typename Tsplits>
template <typename ContainerVals, typename ContainerSplits>
std::pair<std::vector<std::vector<T>>, std::vector<std::vector<Tsplits>>>
RoundRobinTrimmer<T, Tsplits>::TrimInternal(
    const std::vector<ContainerVals>& flat_segments,
    const std::vector<ContainerSplits>& row_splits) const {
  std::pair<std::vector<Segment>, std::vector<RowSplits>> trimmed(
      {std::vector<Segment>(flat_segments.size()),
       std::vector<RowSplits>(row_splits.size())});
  // All row splits start at index 0
  for (int i = 0; i < row_splits.size(); ++i) {
    trimmed.second[i].push_back({0});
  }
  ProcessSplitsByBatch(row_splits,
      [&trimmed, &flat_segments, &row_splits](std::vector<Row> segments) {
    for (int s = 0; s < segments.size(); ++s) {
      Segment* vals = &trimmed.first[s];
      RowSplits* splits = &trimmed.second[s];
      auto start = flat_segments[s].begin() + row_splits[s][splits->size()-1];
      vals->insert(vals->end(), start, start + segments[s].used);
      splits->insert(splits->end(), splits->back() + segments[s].used);
    }
  });
  return trimmed;
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::ProcessBatch(
    std::vector<Row>* segment_sizes_ptr) const {
  std::vector<Row>& segment_sizes = *segment_sizes_ptr;
  int num_segments = segment_sizes.size();

  // Fill all segments to the max that you can - smallest first to largest
  std::sort(segment_sizes.begin(), segment_sizes.end(),
            [] (Row a, Row b) { return a.size < b.size; });
  int filled = 0, sequence_left = max_sequence_length_;
  for (; filled < num_segments &&
      segment_sizes[filled].size * (num_segments - filled) <= sequence_left;
      ++filled) {
    segment_sizes[filled].used = segment_sizes[filled].size;
    sequence_left -= segment_sizes[filled].used;
  }

  // Fill the remaining segments evenly
  if (num_segments > filled) {
    int count = sequence_left / (num_segments - filled);
    for (int i = filled; i < num_segments; ++i) {
      segment_sizes[i].used = count;
    }
    sequence_left -= (num_segments - filled) * count;
  }

  // Finally add the remainder - index order
  std::sort(segment_sizes.begin(), segment_sizes.end(),
            [] (Row a, Row b) { return a.idx < b.idx; });
  for (int i = 0; sequence_left > 0 && i < num_segments; ++i) {
    if (segment_sizes[i].used < segment_sizes[i].size) {
      ++segment_sizes[i].used;
      --sequence_left;
    }
  }
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::ProcessSplitsByBatch(
    const std::vector<RowSplitsSpan>& row_splits,
    std::function<void(std::vector<Row>)> callback) const {
  int num_in_batch = row_splits[0].size() - 1;
  int num_segments = row_splits.size();
  // Process one batch at a time.
  std::vector<Row> segment_sizes(num_segments);
  for (int batch_idx = 0; batch_idx < num_in_batch; ++batch_idx) {
    // First, get size of each row segment.
    for (int s = 0; s < num_segments; ++s) {
      segment_sizes[s].idx = s;
      segment_sizes[s].size =
          row_splits[s][batch_idx + 1] - row_splits[s][batch_idx];
    }
    ProcessBatch(&segment_sizes);

    // Usage of rows computed. Execute callback to process.
    callback(segment_sizes);
  }
}

template <typename T, typename Tsplits>
void RoundRobinTrimmer<T, Tsplits>::ProcessSplitsByBatch(
    const std::vector<std::vector<Tsplits>>& row_splits,
    std::function<void(std::vector<Row>)> callback) const {
  std::vector<RowSplitsSpan> row_splits_span(row_splits.size());
  for (int i = 0; i < row_splits.size(); ++i) {
    row_splits_span[i] = RowSplitsSpan(
        const_cast<Tsplits*>(row_splits[i].data()), row_splits[i].size());
  }
  ProcessSplitsByBatch(row_splits_span, callback);
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_ROUND_ROBIN_TRIMMER_H_
