
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"
#include <algorithm>

namespace ONNX_NAMESPACE { namespace optimization {

struct MergeSqueezes final : public OptimizePass {
  explicit MergeSqueezes()
    : OptimizePass("merge_squeezes", API_TYPE::IR) {
  }

  vector<int64_t> modify_axes(std::vector<int64_t> v1; vector<int64_t> v2)  {
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    int i, j = 0;
    while (v2.at(j) < v1.at(0)) {
      v3.push_back(v2.at(j));
      j++;
    }
    for ( ; i < v1.size() - 1; i++) {
      v3.push_back(v1.at(i));
      while (j < v2.size() && v2.at(j) + i + 1 < v1.at(i + 1)) {
        v3.push_back(v2.at(j) + i + 1);
        j++;
      }
    }
    v3.push_back(v1.at(i));
    while (j < v2.size()) {
      v3.push_back(v2.at(j) + i + 1);
      j++;
    }
    return v3;
  }

  void merge_squeezes(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){merge_squeezes(g);});
      if (n->kind() == kSqueeze) {
        auto squeeze_input = n->input();
        if (squeeze_input->node().kind() != kSqueeze || squeeze_input.uses().size() != 1)
          continue;
        n->replaceInput(0, squeeze_input->node()->input());
        n->is_(kaxes, modify_axes(squeeze_input->is(kaxes), n->is(kaxes)));
        squeeze_input->node()->destroy();
      }
    }
  }

  void optimize(Graph& graph) override {
    merge_squeezes(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
