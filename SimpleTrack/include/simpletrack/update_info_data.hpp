// UpdateInfoData translated from mot_3d/update_info_data.py
#pragma once

#include <Eigen/Dense>

#include <vector>

#include "simpletrack/bbox.hpp"
#include "simpletrack/frame_data.hpp"

namespace simpletrack {

struct UpdateInfoData {
    int mode{0};
    BBox bbox;
    int frame_index{0};
    Eigen::Matrix4d ego{Eigen::Matrix4d::Identity()};
    std::vector<BBox> dets;
    std::vector<Eigen::Vector3d> pc;
    AuxInfo aux_info;

    UpdateInfoData() = default;
    UpdateInfoData(int mode_in,
                   const BBox& bbox_in,
                   int frame_index_in,
                   const Eigen::Matrix4d& ego_in,
                   AuxInfo aux_info_in = AuxInfo(),
                   std::vector<BBox> dets_in = {},
                   std::vector<Eigen::Vector3d> pc_in = {});
};

}  // namespace simpletrack

