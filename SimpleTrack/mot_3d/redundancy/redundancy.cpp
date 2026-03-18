#include "simpletrack/redundancy.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>

#include "simpletrack/association.hpp"
#include "simpletrack/geometry.hpp"

namespace simpletrack {

namespace {

double lookup_threshold(const std::unordered_map<std::string, double>& table,
                         const std::string& key,
                         double fallback) {
    auto it = table.find(key);
    if (it != table.end()) {
        return it->second;
    }
    return fallback;
}

}

RedundancyModule::RedundancyModule(const TrackerConfig& configs)
    : configs_(configs),
      mode_(configs.redundancy.mode),
      asso_(configs.running.asso) {
    det_score_ = lookup_threshold(configs.redundancy.det_score_threshold, asso_, 0.0);
    det_threshold_ = lookup_threshold(configs.redundancy.det_dist_threshold, asso_, 0.0);
}

RedundancyResult RedundancyModule::default_redundancy(Tracklet& trk, const FrameData&) {
    RedundancyResult result;
    result.bbox = trk.get_state();
    result.update_mode = 0;
    result.velo.setZero();
    return result;
}

RedundancyResult RedundancyModule::motion_model_redundancy(Tracklet& trk,
                                                           const FrameData& frame,
                                                           double /*time_lag*/) {
    RedundancyResult result;
    result.bbox = trk.get_state();
    result.velo.setZero();

    std::vector<double> distances;
    distances.reserve(frame.dets.size());
    bool need_inversion = (asso_ == "m_dis");
    std::optional<Eigen::Matrix<double, 7, 7>> inv_innovation;
    for (const auto& det : frame.dets) {
        if (!det.has_score() || det.s > det_score_) {
            double dist = 0.0;
            if (asso_ == "iou") {
                dist = iou3d(det, result.bbox).second;
            } else if (asso_ == "giou") {
                dist = giou3d(det, result.bbox);
            } else if (asso_ == "m_dis") {
                if (!inv_innovation.has_value()) {
                    inv_innovation = trk.compute_innovation_matrix().inverse();
                }
                dist = m_distance(det, result.bbox, *inv_innovation);
            } else if (asso_ == "euler") {
                dist = m_distance(det, result.bbox, std::nullopt);
            }
            distances.push_back(dist);
        }
    }

    if (asso_ == "iou" || asso_ == "giou") {
        const double max_val = distances.empty() ? 0.0 : *std::max_element(distances.begin(), distances.end());
        result.update_mode = (distances.empty() || max_val < det_threshold_) ? 0 : 3;
    } else {
        const double min_val = distances.empty() ? std::numeric_limits<double>::infinity()
                                                 : *std::min_element(distances.begin(), distances.end());
        result.update_mode = (distances.empty() || min_val > det_threshold_) ? 0 : 3;
    }
    return result;
}

RedundancyResult RedundancyModule::infer(Tracklet& trk,
                                         const FrameData& frame,
                                         double time_lag) {
    if (mode_ == "bbox") {
        return default_redundancy(trk, frame);
    }
    if (mode_ == "mm") {
        return motion_model_redundancy(trk, frame, time_lag);
    }
    return default_redundancy(trk, frame);
}

std::pair<std::vector<BBox>, std::vector<int>> RedundancyModule::bipartite_infer(
    const FrameData& frame,
    std::vector<std::unique_ptr<Tracklet>>& tracklets) {
    std::vector<int> det_indexes;
    det_indexes.reserve(frame.dets.size());
    for (int i = 0; i < static_cast<int>(frame.dets.size()); ++i) {
        const auto& det = frame.dets[i];
        if (!det.has_score() || det.s >= det_score_) {
            det_indexes.push_back(i);
        }
    }

    std::vector<BBox> dets_filtered;
    dets_filtered.reserve(det_indexes.size());
    for (int idx : det_indexes) {
        dets_filtered.push_back(frame.dets[idx]);
    }

    std::vector<BBox> predictions;
    predictions.reserve(tracklets.size());
    for (auto& trk : tracklets) {
        predictions.push_back(trk->predict(frame.time_stamp, frame.aux_info.is_key_frame));
    }

    AssociationResult assoc = associate_dets_to_tracks(
        dets_filtered,
        predictions,
        "bipartite",
        "giou",
        1.0 - det_threshold_,
        nullptr);

    for (auto& match : assoc.matches) {
        if (match.first >= 0 && match.first < static_cast<int>(det_indexes.size())) {
            match.first = det_indexes[match.first];
        }
    }
    for (auto& det_idx : assoc.unmatched_dets) {
        if (det_idx >= 0 && det_idx < static_cast<int>(det_indexes.size())) {
            det_idx = det_indexes[det_idx];
        }
    }

    std::vector<BBox> result_bboxes;
    std::vector<int> update_modes;
    result_bboxes.reserve(tracklets.size());
    update_modes.reserve(tracklets.size());

    for (size_t t = 0; t < tracklets.size(); ++t) {
        bool matched = false;
        for (const auto& match : assoc.matches) {
            if (match.second == static_cast<int>(t)) {
                matched = true;
                break;
            }
        }
        result_bboxes.push_back(predictions[t]);
        update_modes.push_back(matched ? 4 : 0);
    }

    return {result_bboxes, update_modes};
}

}  // namespace simpletrack
