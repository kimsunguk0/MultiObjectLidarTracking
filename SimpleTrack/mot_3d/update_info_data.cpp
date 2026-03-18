#include "simpletrack/update_info_data.hpp"

#include <utility>

namespace simpletrack {

UpdateInfoData::UpdateInfoData(int mode_in,
                               const BBox& bbox_in,
                               int frame_index_in,
                               const Eigen::Matrix4d& ego_in,
                               AuxInfo aux_info_in,
                               std::vector<BBox> dets_in,
                               std::vector<Eigen::Vector3d> pc_in)
    : mode(mode_in),
      bbox(bbox_in),
      frame_index(frame_index_in),
      ego(ego_in),
      dets(std::move(dets_in)),
      pc(std::move(pc_in)),
      aux_info(std::move(aux_info_in)) {}

}  // namespace simpletrack
