// FrameData structure translated from mot_3d/frame_data.py
#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

struct AuxInfo {
    bool is_key_frame{true};
    std::vector<Eigen::Vector3d> velos;
    bool has_velos{false};
    bool has_cls_name{false};
    std::string cls_name;
};

struct FrameData {
    std::vector<BBox> dets;
    Eigen::Matrix4d ego{Eigen::Matrix4d::Identity()};
    double time_stamp{0.0};
    std::vector<Eigen::Vector3d> pc;
    std::vector<int> det_types;
    AuxInfo aux_info;

    FrameData() = default;
    FrameData(std::vector<BBox> dets_in,
              const Eigen::Matrix4d& ego_in,
              double time_stamp_in,
              std::vector<int> det_types_in,
              AuxInfo aux_info_in = AuxInfo(),
              std::vector<Eigen::Vector3d> pc_in = {});
};

}  // namespace simpletrack

