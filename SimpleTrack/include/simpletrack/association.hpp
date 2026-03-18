// Association utilities translated from mot_3d/association.py
#pragma once

#include <Eigen/Dense>

#include <string>
#include <utility>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

struct AssociationResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_dets;
    std::vector<int> unmatched_tracks;
};

AssociationResult associate_dets_to_tracks(
    const std::vector<BBox>& dets,
    const std::vector<BBox>& tracks,
    const std::string& mode,
    const std::string& asso,
    double dist_threshold,
    const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation_matrix = nullptr);

Eigen::MatrixXd compute_iou_distance(const std::vector<BBox>& dets,
                                     const std::vector<BBox>& tracks,
                                     const std::string& asso);

Eigen::MatrixXd compute_m_distance(const std::vector<BBox>& dets,
                                   const std::vector<BBox>& tracks,
                                   const std::vector<Eigen::Matrix<double, 7, 7>>* trk_innovation);

}  // namespace simpletrack
