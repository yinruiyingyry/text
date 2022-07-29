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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_

#include <vector>

#include "absl/types/span.h"

namespace tensorflow {
namespace text {

template <typename T, typename Tsplits = int32_t>
class Trimmer {
  using Mask = std::vector<bool>;
  using Segment = std::vector<T>;
  using SegmentSpan = absl::Span<T>;
  using RowSplits = std::vector<Tsplits>;
  using RowSplitsSpan = absl::Span<Tsplits>;

 public:
  // Generates masks for a single batch of segments.
  virtual std::vector<Mask> GenerateMasks(
      const std::vector<Segment>& segments) const;

  // Generates masks for a batch of segment row splits.
  //
  // The returned value is a flattened list of mask values which can be split
  // into batches using the same input row splits.
  virtual std::vector<Mask> GenerateMaskBatches(
      const std::vector<RowSplits>& row_splits) const;
  virtual std::vector<Mask> GenerateMaskBatches(
      const std::vector<RowSplitsSpan>& row_splits) const;

  // Trims a single batch of segments.
  virtual void Trim(std::vector<Segment>* segments) const;

  // Trims a batch of segments given their flattened values and row splits.
  //
  // The returned values are the flattened trimmed values and new row splits.
  virtual std::pair<std::vector<Segment>, std::vector<RowSplits>> Trim(
      const std::vector<Segment>& flat_segments,
      const std::vector<RowSplits>& row_splits) const;
  virtual std::pair<std::vector<Segment>, std::vector<RowSplits>> Trim(
      const std::vector<SegmentSpan>& flat_segments,
      const std::vector<RowSplitsSpan>& row_splits) const;

  virtual ~Trimmer() = default;
};

}  // namespace text
}  // namespace tensorflow


#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_TRIMMER_H_
