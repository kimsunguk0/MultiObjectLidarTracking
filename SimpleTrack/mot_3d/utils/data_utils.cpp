#include "simpletrack/data_utils.hpp"

#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <vector>

namespace simpletrack {

std::vector<int> str2int(const std::vector<std::string>& strs) {
    std::vector<int> result;
    result.reserve(strs.size());
    for (const auto& s : strs) {
        result.push_back(std::stoi(s));
    }
    return result;
}

std::vector<std::vector<std::pair<int, BBox>>> box_wrapper(
    const std::vector<std::vector<BBox>>& bboxes,
    const std::vector<std::vector<int>>& ids) {
    if (bboxes.size() != ids.size()) {
        throw std::invalid_argument("bboxes and ids must have identical frame counts");
    }
    std::vector<std::vector<std::pair<int, BBox>>> result;
    result.resize(ids.size());
    for (size_t frame = 0; frame < ids.size(); ++frame) {
        const auto& frame_ids = ids[frame];
        const auto& frame_boxes = bboxes[frame];
        if (frame_ids.size() != frame_boxes.size()) {
            throw std::invalid_argument("ids and bboxes must match within each frame");
        }
        auto& dst = result[frame];
        dst.reserve(frame_ids.size());
        for (size_t i = 0; i < frame_ids.size(); ++i) {
            dst.emplace_back(frame_ids[i], frame_boxes[i]);
        }
    }
    return result;
}

std::vector<std::vector<int>> id_transform(const std::vector<std::vector<int>>& ids) {
    std::unordered_map<int, int> id_mapping;
    for (const auto& frame_ids : ids) {
        for (int id : frame_ids) {
            id_mapping.emplace(id, 0);
        }
    }

    int index = 0;
    for (auto& kv : id_mapping) {
        kv.second = index++;
    }

    std::vector<std::vector<int>> result(ids.size());
    for (size_t frame = 0; frame < ids.size(); ++frame) {
        const auto& frame_ids = ids[frame];
        auto& dst = result[frame];
        dst.reserve(frame_ids.size());
        for (int id : frame_ids) {
            dst.push_back(id_mapping.at(id));
        }
    }
    return result;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<BBox>>> inst_filter(
    const std::vector<std::vector<int>>& ids,
    const std::vector<std::vector<std::vector<double>>>& bboxes,
    const std::vector<std::vector<int>>& types,
    const std::vector<int>& type_field,
    bool id_trans) {
    if (ids.size() != bboxes.size() || ids.size() != types.size()) {
        throw std::invalid_argument("ids, bboxes, and types must share the same frame length");
    }

    std::vector<std::vector<int>> processed_ids = ids;
    if (id_trans) {
        processed_ids = id_transform(ids);
    }

    std::vector<std::vector<int>> id_result;
    std::vector<std::vector<BBox>> bbox_result;
    id_result.resize(ids.size());
    bbox_result.resize(ids.size());

    for (size_t frame = 0; frame < ids.size(); ++frame) {
        const auto& frame_ids = processed_ids[frame];
        const auto& frame_bboxes = bboxes[frame];
        const auto& frame_types = types[frame];
        if (frame_ids.size() != frame_bboxes.size() || frame_ids.size() != frame_types.size()) {
            throw std::invalid_argument("Frame data mismatch in inst_filter");
        }

        auto& dst_ids = id_result[frame];
        auto& dst_boxes = bbox_result[frame];

        for (size_t i = 0; i < frame_ids.size(); ++i) {
            int obj_type = frame_types[i];
            bool matched = false;
            for (int type_name : type_field) {
                if (type_name == obj_type) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                continue;
            }

            const auto& arr = frame_bboxes[i];
            if (arr.size() < 7) {
                throw std::invalid_argument("bbox array must contain at least 7 elements");
            }
            std::vector<double> buffer(arr.begin(), arr.end());
            dst_ids.push_back(frame_ids[i]);
            dst_boxes.push_back(BBox::from_vector(buffer));
        }
    }

    return {id_result, bbox_result};
}

std::vector<std::vector<std::vector<int>>> type_filter(
    const std::vector<std::vector<std::vector<int>>>& contents,
    const std::vector<std::vector<int>>& types,
    const std::vector<int>& type_field) {
    throw std::logic_error("type_filter not implemented in C++ port");
    (void)contents;
    (void)types;
    (void)type_field;
    return {};
}

}  // namespace simpletrack
