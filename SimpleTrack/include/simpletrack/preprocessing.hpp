#pragma once

#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

class BBoxCoarseFilter {
public:
    explicit BBoxCoarseFilter(int grid_size, int scaler = 100)
        : grid_size_(grid_size), scaler_(scaler) {}

    void bboxes2dict(const std::vector<BBox>& bboxes);
    std::vector<int> related_bboxes(const BBox& bbox) const;
    void clear();

private:
    std::vector<int> compute_bbox_key(const BBox& bbox) const;

    int grid_size_{100};
    int scaler_{100};
    std::unordered_map<int, std::unordered_set<int>> bbox_dict_;
};

bool weird_bbox(const BBox& bbox);

std::pair<std::vector<int>, std::vector<int>> nms(
    const std::vector<BBox>& dets,
    const std::vector<int>& inst_types,
    double threshold_low = 0.1,
    double threshold_high = 1.0,
    double threshold_yaw = 0.3);

}  // namespace simpletrack

