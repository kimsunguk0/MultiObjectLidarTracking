// Tracker configuration structures for the C++ port.
#pragma once

#include <Eigen/Dense>

#include <optional>
#include <string>
#include <unordered_map>

namespace simpletrack {

struct RunningConfig {
    std::string match_type{"bipartite"};
    std::string asso{"giou"};
    std::string motion_model{"kf"};
    double score_threshold{0.0};
    double post_nms_iou{0.0};
    int max_age_since_update{3};
    int min_hits_to_birth{3};
    std::unordered_map<std::string, double> asso_thresholds;
    std::optional<Eigen::Matrix<double, 7, 7>> measurement_noise;
    std::optional<Eigen::Matrix<double, 10, 10>> covariance;
};

struct RedundancyConfig {
    std::string mode{"mm"};
    std::unordered_map<std::string, double> det_score_threshold;
    std::unordered_map<std::string, double> det_dist_threshold;
};

struct TrackerConfig {
    RunningConfig running;
    RedundancyConfig redundancy;
    // Additional sections (data_loader, visualization, etc.) can be added as needed.
};

}  // namespace simpletrack
