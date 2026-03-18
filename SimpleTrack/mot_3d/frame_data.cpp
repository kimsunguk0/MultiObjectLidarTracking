#include "simpletrack/frame_data.hpp"

#include <utility>

namespace simpletrack {

FrameData::FrameData(std::vector<BBox> dets_in,
                     const Eigen::Matrix4d& ego_in,
                     double time_stamp_in,
                     std::vector<int> det_types_in,
                     AuxInfo aux_info_in,
                     std::vector<Eigen::Vector3d> pc_in)
    : dets(std::move(dets_in)),
      ego(ego_in),
      time_stamp(time_stamp_in),
      pc(std::move(pc_in)),
      det_types(std::move(det_types_in)),
      aux_info(std::move(aux_info_in)) {}

}  // namespace simpletrack
