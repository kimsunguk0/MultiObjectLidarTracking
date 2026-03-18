#include "simpletrack/preprocessing.hpp"

#include <cmath>

namespace simpletrack {

void BBoxCoarseFilter::bboxes2dict(const std::vector<BBox>& bboxes) {
    clear();
    for (int i = 0; i < static_cast<int>(bboxes.size()); ++i) {
        auto keys = compute_bbox_key(bboxes[i]);
        for (int key : keys) {
            bbox_dict_[key].insert(i);
        }
    }
}

std::vector<int> BBoxCoarseFilter::related_bboxes(const BBox& bbox) const {
    std::unordered_set<int> result;
    auto keys = compute_bbox_key(bbox);
    for (int key : keys) {
        auto it = bbox_dict_.find(key);
        if (it != bbox_dict_.end()) {
            result.insert(it->second.begin(), it->second.end());
        }
    }
    return std::vector<int>(result.begin(), result.end());
}

void BBoxCoarseFilter::clear() {
    bbox_dict_.clear();
}

std::vector<int> BBoxCoarseFilter::compute_bbox_key(const BBox& bbox) const {
    auto corners = box2corners2d(bbox);
    double min_x = corners[0][0], max_x = corners[0][0];
    double min_y = corners[0][1], max_y = corners[0][1];
    for (const auto& c : corners) {
        min_x = std::min(min_x, c[0]);
        max_x = std::max(max_x, c[0]);
        min_y = std::min(min_y, c[1]);
        max_y = std::max(max_y, c[1]);
    }
    int min_key_x = static_cast<int>(std::floor(min_x / grid_size_));
    int max_key_x = static_cast<int>(std::floor(max_x / grid_size_));
    int min_key_y = static_cast<int>(std::floor(min_y / grid_size_));
    int max_key_y = static_cast<int>(std::floor(max_y / grid_size_));

    std::vector<int> keys;
    keys.reserve(4);
    keys.push_back(scaler_ * min_key_x + min_key_y);
    keys.push_back(scaler_ * min_key_x + max_key_y);
    keys.push_back(scaler_ * max_key_x + min_key_y);
    keys.push_back(scaler_ * max_key_x + max_key_y);
    return keys;
}

}  // namespace simpletrack

