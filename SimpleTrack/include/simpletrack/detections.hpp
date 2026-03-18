#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "simpletrack/boundary.hpp"
#include "simpletrack/frame_data.hpp"

namespace simpletrack {

struct RawDetection {
    double score{0.0};
    double x_px{0.0};
    double y_px{0.0};
    double z_m{0.0};
    double h_m{0.0};
    double w_px{0.0};
    double l_px{0.0};
    double yaw_rad{0.0};
    int cls{0};
};

FrameData detections_to_framedata(const std::vector<RawDetection>& detections,
                                  double timestamp,
                                  const Eigen::Matrix4d& ego,
                                  bool is_key_frame,
                                  const std::string& cls_name,
                                  const Boundary& boundary = kDefaultBoundary);

}  // namespace simpletrack

