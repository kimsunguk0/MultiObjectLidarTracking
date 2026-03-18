// Redundancy module translated from mot_3d/redundancy/redundancy.py
#pragma once

#include <string>
#include <vector>

#include "simpletrack/config.hpp"
#include "simpletrack/frame_data.hpp"
#include "simpletrack/tracklet.hpp"

namespace simpletrack {

struct RedundancyResult {
    BBox bbox;
    int update_mode{0};
    Eigen::Vector2d velo{Eigen::Vector2d::Zero()};
};

class RedundancyModule {
public:
    explicit RedundancyModule(const TrackerConfig& configs);

    RedundancyResult infer(Tracklet& trk, const FrameData& frame, double time_lag = 0.0);

    std::pair<std::vector<BBox>, std::vector<int>> bipartite_infer(
        const FrameData& frame,
        std::vector<std::unique_ptr<Tracklet>>& tracklets);

private:
    RedundancyResult default_redundancy(Tracklet& trk, const FrameData& frame);
    RedundancyResult motion_model_redundancy(Tracklet& trk, const FrameData& frame, double time_lag);

    TrackerConfig configs_;
    std::string mode_;
    std::string asso_;
    double det_score_{0.0};
    double det_threshold_{0.0};
};

}  // namespace simpletrack

