// Utility helpers translated from mot_3d/utils/data_utils.py
#pragma once

#include <string>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

std::vector<int> str2int(const std::vector<std::string>& strs);

std::vector<std::vector<std::pair<int, BBox>>> box_wrapper(
    const std::vector<std::vector<BBox>>& bboxes,
    const std::vector<std::vector<int>>& ids);

std::vector<std::vector<int>> id_transform(const std::vector<std::vector<int>>& ids);

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<BBox>>> inst_filter(
    const std::vector<std::vector<int>>& ids,
    const std::vector<std::vector<std::vector<double>>>& bboxes,
    const std::vector<std::vector<int>>& types,
    const std::vector<int>& type_field,
    bool id_trans);

std::vector<std::vector<std::vector<int>>> type_filter(
    const std::vector<std::vector<std::vector<int>>>& contents,
    const std::vector<std::vector<int>>& types,
    const std::vector<int>& type_field);

}  // namespace simpletrack
