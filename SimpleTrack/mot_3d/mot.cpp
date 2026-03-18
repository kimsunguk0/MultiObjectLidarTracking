#include "simpletrack/mot_model.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <chrono>

#include "simpletrack/association.hpp"
#include "simpletrack/geometry.hpp"
#include "simpletrack/redundancy.hpp"
#include "simpletrack/update_info_data.hpp"

namespace simpletrack {

namespace {

double lookup_threshold(const TrackerConfig& configs) {
    auto it = configs.running.asso_thresholds.find(configs.running.asso);
    if (it != configs.running.asso_thresholds.end()) {
        return it->second;
    }
    return 1.0;
}

std::string format_bbox(const BBox& bbox) {
    std::ostringstream oss;
    double score = bbox.has_score() ? bbox.s : std::numeric_limits<double>::quiet_NaN();
    oss << std::fixed << std::setprecision(2)
        << "(x=" << bbox.x
        << ", y=" << bbox.y
        << ", z=" << bbox.z
        << ", l=" << bbox.l
        << ", w=" << bbox.w
        << ", h=" << bbox.h
        << ", o=" << bbox.o
        << ", s=" << score << ")";
    return oss.str();
}

int detection_type_at(const FrameData& frame, int index) {
    if (index >= 0 && index < static_cast<int>(frame.det_types.size())) {
        return frame.det_types[index];
    }
    return 0;
}

}  // namespace

MOTModel::MOTModel(TrackerConfig configs)
    : configs_(std::move(configs)),
      redundancy_(configs_) {}

void MOTModel::set_debug(bool association, bool lifecycle) {
    debug_association_ = association;
    debug_lifecycle_ = lifecycle;
}

std::vector<BBox> MOTModel::predict_tracks(double time_stamp, bool is_key_frame) {
    std::vector<BBox> predictions;
    predictions.reserve(trackers_.size());
    for (auto& trk : trackers_) {
        predictions.push_back(trk->predict(time_stamp, is_key_frame));
    }
    return predictions;
}

void MOTModel::update_matched_tracks(const FrameData& frame,
                                     const std::vector<std::pair<int, int>>& matches,
                                     const std::vector<int>& det_indexes,
                                     const Eigen::MatrixXd* dist_matrix) {
    for (const auto& match : matches) {
        const int det_local = match.first;
        const int trk_idx = match.second;
        if (trk_idx < 0 || trk_idx >= static_cast<int>(trackers_.size())) {
            continue;
        }
        if (det_local < 0 || det_local >= static_cast<int>(det_indexes.size())) {
            continue;
        }
        const int det_global = det_indexes[det_local];
        if (det_global < 0 || det_global >= static_cast<int>(frame.dets.size())) {
            continue;
        }
        auto& track = trackers_[trk_idx];
        double det_score = frame.dets[det_global].has_score() ? frame.dets[det_global].s : std::numeric_limits<double>::quiet_NaN();
        double dist_val = std::numeric_limits<double>::quiet_NaN();
        if (dist_matrix && det_local >= 0 && det_local < dist_matrix->rows() && trk_idx < dist_matrix->cols()) {
            dist_val = (*dist_matrix)(det_local, trk_idx);
        }
        std::string dist_msg = "NA";
        if (!std::isnan(dist_val)) {
            std::ostringstream ds;
            ds << std::fixed << std::setprecision(3) << dist_val;
            dist_msg = ds.str();
        }
        std::string pre_state_bbox = format_bbox(track->motion_model().get_state());
        const auto& lm_pre = track->life_manager();
        if (debug_lifecycle_) {
            std::cout << "  [Update] track#" << track->id() << " \u2190 det#" << det_global
                      << " score=" << std::fixed << std::setprecision(2) << det_score
                      << " dist=" << dist_msg
                      << " state=" << lm_pre.state << ", tsu=" << lm_pre.time_since_update << "\n";
        }
        AuxInfo aux = frame.aux_info;
        UpdateInfoData update_info(
            /*mode*/ 1,
            frame.dets[det_global],
            frame_count_,
            frame.ego,
            aux,
            frame.dets,
            frame.pc);
        track->update(update_info);
        if (debug_lifecycle_) {
            const auto& lm_post = track->life_manager();
            std::cout << "    -> post-update state=" << lm_post.state
                      << ", hits=" << lm_post.hits
                      << ", tsu=" << lm_post.time_since_update << "\n";
            std::string post_state_bbox = format_bbox(track->motion_model().get_state());
            std::cout << "    -> KF state change: " << pre_state_bbox
                      << "  ==>  " << post_state_bbox << "\n";
            std::cout << std::defaultfloat;
        }
    }
}

void MOTModel::update_unmatched_tracks(const FrameData& frame,
                                       const std::vector<int>& unmatched_tracks,
                                       const std::vector<BBox>& predicted,
                                       const Eigen::MatrixXd* dist_matrix) {
    for (int trk_idx : unmatched_tracks) {
        if (trk_idx < 0 || trk_idx >= static_cast<int>(trackers_.size())) {
            continue;
        }
        auto& track = trackers_[trk_idx];
        RedundancyResult redundancy = redundancy_.infer(*track, frame, frame.time_stamp - last_time_stamp_);
        AuxInfo aux = frame.aux_info;
        if (debug_lifecycle_) {
            double best_dist = std::numeric_limits<double>::infinity();
            int best_det = -1;
            if (dist_matrix && dist_matrix->rows() > 0 && trk_idx < dist_matrix->cols()) {
                for (int r = 0; r < dist_matrix->rows(); ++r) {
                    double d = (*dist_matrix)(r, trk_idx);
                    if (d < best_dist) {
                        best_dist = d;
                        best_det = r;
                    }
                }
            }
            const auto& lm_pre = track->life_manager();
            std::cout << "  [Update] track#" << track->id() << " unmatched -> redundancy "
                      << "mode=" << redundancy.update_mode
                      << ", state=" << lm_pre.state
                      << ", tsu=" << lm_pre.time_since_update;
            if (best_det >= 0) {
                std::cout << " | closest det idx=" << best_det
                          << " dist=" << std::fixed << std::setprecision(3) << best_dist;
            }
            std::cout << "\n";
        }
        UpdateInfoData update_info(
            /*mode*/ redundancy.update_mode,
            redundancy.bbox,
            frame_count_,
            frame.ego,
            aux,
            frame.dets,
            frame.pc);
        track->update(update_info);
        if (debug_lifecycle_) {
            const auto& lm_post = track->life_manager();
            std::cout << "    -> post-update state=" << lm_post.state
                      << ", hits=" << lm_post.hits
                      << ", tsu=" << lm_post.time_since_update << "\n";
            std::cout << std::defaultfloat;
        }
    }
}

void MOTModel::spawn_tracks(const FrameData& frame,
                            const std::vector<int>& unmatched_dets) {
    bool enable_debug = debug_association_ && debug_lifecycle_;  // effectively debug-all
    for (int det_idx : unmatched_dets) {
        if (det_idx < 0 || det_idx >= static_cast<int>(frame.dets.size())) {
            continue;
        }
        int det_type = detection_type_at(frame, det_idx);
        auto trk = std::make_unique<Tracklet>(
            configs_,
            next_track_id_++,
            frame.dets[det_idx],
            det_type,
            frame_count_,
            frame.time_stamp,
            frame.aux_info,
            enable_debug);
        if (debug_lifecycle_) {
            std::cout << "  [Spawn] track#" << trk->id()
                      << " from det#" << det_idx
                      << " type=" << det_type << " "
                      << format_bbox(frame.dets[det_idx]) << "\n";
        }
        trackers_.push_back(std::move(trk));
    }
}

void MOTModel::prune_tracks(bool log_active_tracks) {
    auto it = trackers_.begin();
    while (it != trackers_.end()) {
        if ((*it)->death(frame_count_)) {
            if (log_active_tracks) {
                const auto& lm = (*it)->life_manager();
                std::cout << "  [Death] track#" << (*it)->id()
                          << " removed (state=" << lm.state
                          << ", hits=" << lm.hits
                          << ", tsu=" << lm.time_since_update << ")\n";
            }
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }
}

static void apply_post_nms(const TrackerConfig& configs,
                           int frame_count,
                           std::vector<std::unique_ptr<Tracklet>>& trackers,
                           bool log_event) {
    double post_nms_iou = configs.running.post_nms_iou;
    if (post_nms_iou <= 0.0 || trackers.size() <= 1) {
        return;
    }
    std::vector<int> order(trackers.size());
    for (size_t i = 0; i < trackers.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::sort(order.begin(), order.end(),
              [&](int lhs, int rhs) {
                  return trackers[lhs]->compute_innovation_matrix().trace() <
                         trackers[rhs]->compute_innovation_matrix().trace();
              });

    for (int idx : order) {
        auto& base = trackers[idx];
        const BBox base_bbox = base->get_state();
        const double base_score = base_bbox.has_score() ? base_bbox.s : 0.0;
        for (size_t other_idx = 0; other_idx < trackers.size(); ++other_idx) {
            if (static_cast<int>(other_idx) == idx) {
                continue;
            }
            auto& other = trackers[other_idx];
            if (other->det_type() != base->det_type()) {
                continue;
            }
            const BBox other_bbox = other->get_state();
            const double iou3d_val = iou3d(base_bbox, other_bbox).second;
            if (iou3d_val >= post_nms_iou) {
                if (log_event) {
                    std::cout << "  [PostNMS] track#" << other->id()
                              << " (age=" << other->life_manager().time_since_update << ") \u2190 track#"
                              << base->id() << " (age=" << base->life_manager().time_since_update
                              << ") iou=" << std::fixed << std::setprecision(3) << iou3d_val << "\n";
                    std::cout << "           " << format_bbox(other_bbox)
                              << " ==> " << format_bbox(base_bbox) << "\n";
                    std::cout << std::defaultfloat;
                }
                AuxInfo dummy_aux;
                UpdateInfoData overwrite_info(
                    /*mode*/ 3,
                    base_bbox,
                    frame_count,
                    Eigen::Matrix4d::Identity(),
                    dummy_aux);
                other->update(overwrite_info);
            }
        }
    }
}

std::vector<TrackResult> MOTModel::frame_mot(const FrameData& frame, TrackTiming* timing) {
    frame_count_ += 1;
    if (!has_time_stamp_) {
        last_time_stamp_ = frame.time_stamp;
        has_time_stamp_ = true;
    }

    const bool is_key_frame = frame.aux_info.is_key_frame;

    if (debug_association_ || debug_lifecycle_) {
        std::ostringstream tag;
        tag << "[Frame " << frame_count_ << "]";
        if (frame.aux_info.has_cls_name) {
            tag << "[" << frame.aux_info.cls_name << "]";
        }
        std::cout << "\n" << tag.str()
                  << " time=" << std::fixed << std::setprecision(3) << frame.time_stamp
                  << " key=" << (is_key_frame ? "True" : "False")
                  << " dets=" << frame.dets.size()
                  << " active_tracks=" << trackers_.size() << "\n";
        std::cout << std::defaultfloat;
    }

    std::vector<int> det_indexes;
    det_indexes.reserve(frame.dets.size());
    std::vector<int> dropped_det_indexes;
    if (debug_association_) {
        dropped_det_indexes.reserve(frame.dets.size());
    }
    for (size_t i = 0; i < frame.dets.size(); ++i) {
        const double score = frame.dets[i].has_score() ? frame.dets[i].s : 1.0;
        if (score >= configs_.running.score_threshold) {
            det_indexes.push_back(static_cast<int>(i));
        } else if (debug_association_) {
            dropped_det_indexes.push_back(static_cast<int>(i));
        }
    }
    if (debug_association_ && !dropped_det_indexes.empty()) {
        std::cout << "  [Assoc-key] Dropped detections (score < "
                  << std::fixed << std::setprecision(2) << configs_.running.score_threshold
                  << "):\n";
        for (int idx : dropped_det_indexes) {
            if (idx >= 0 && idx < static_cast<int>(frame.dets.size())) {
                std::cout << "    det#" << idx << " " << format_bbox(frame.dets[idx]) << "\n";
            }
        }
        std::cout << std::defaultfloat;
    }

    std::vector<BBox> dets_filtered;
    dets_filtered.reserve(det_indexes.size());
    for (int idx : det_indexes) {
        dets_filtered.push_back(frame.dets[idx]);
    }

    auto t_predict_start = std::chrono::high_resolution_clock::now();
    std::vector<BBox> trk_predictions = predict_tracks(frame.time_stamp, is_key_frame);
    auto t_predict_end = std::chrono::high_resolution_clock::now();

    auto t_assoc_start = t_predict_end;
    std::vector<Eigen::Matrix<double, 7, 7>> innovation_matrices;
    std::vector<Eigen::Matrix<double, 7, 7>>* innovation_ptr = nullptr;
    if (configs_.running.asso == "m_dis") {
        innovation_matrices.reserve(trackers_.size());
        for (const auto& trk : trackers_) {
            innovation_matrices.push_back(trk->compute_innovation_matrix());
        }
        innovation_ptr = &innovation_matrices;
    }

    const double dist_threshold = lookup_threshold(configs_);
    AssociationResult assoc = associate_dets_to_tracks(
        dets_filtered,
        trk_predictions,
        configs_.running.match_type,
        configs_.running.asso,
        dist_threshold,
        innovation_ptr);
    auto t_assoc_end = std::chrono::high_resolution_clock::now();

    auto t_dist_start = t_assoc_end;
    Eigen::MatrixXd dist_matrix;
    const Eigen::MatrixXd* dist_ptr = nullptr;
    if (!dets_filtered.empty() && !trk_predictions.empty()) {
        if (configs_.running.asso == "m_dis") {
            dist_matrix = compute_m_distance(dets_filtered, trk_predictions, innovation_ptr);
        } else if (configs_.running.asso == "euler") {
            dist_matrix = compute_m_distance(dets_filtered, trk_predictions, nullptr);
        } else {
            dist_matrix = compute_iou_distance(dets_filtered, trk_predictions, configs_.running.asso);
        }
        dist_ptr = &dist_matrix;
    }
    auto t_dist_end = std::chrono::high_resolution_clock::now();

    if (debug_association_ || debug_lifecycle_) {
        std::cout << "  [Assoc-key] Matches: " << assoc.matches.size()
                  << ", Unmatched dets: " << assoc.unmatched_dets.size()
                  << ", Unmatched trks: " << assoc.unmatched_tracks.size() << "\n";
        if (debug_association_) {
            if (!assoc.matches.empty()) {
                for (const auto& match : assoc.matches) {
                    int det_local = match.first;
                    int trk_idx = match.second;
                    if (det_local < 0 || det_local >= static_cast<int>(det_indexes.size()) ||
                        trk_idx < 0 || trk_idx >= static_cast<int>(trackers_.size())) {
                        continue;
                    }
                    int det_global = det_indexes[det_local];
                    const BBox& det_box = frame.dets[det_global];
                    const BBox& pred_box = trk_predictions[trk_idx];
                    const auto& lm = trackers_[trk_idx]->life_manager();
                    double dist_val = std::numeric_limits<double>::quiet_NaN();
                    if (dist_ptr && det_local < dist_ptr->rows() && trk_idx < dist_ptr->cols()) {
                        dist_val = (*dist_ptr)(det_local, trk_idx);
                    }
                    std::ostringstream dist_stream;
                    if (!std::isnan(dist_val)) {
                        dist_stream << std::fixed << std::setprecision(3) << dist_val;
                    } else {
                        dist_stream << "NA";
                    }
                    std::cout << "    det#" << det_global << " " << format_bbox(det_box)
                              << " \u2192 track#" << trackers_[trk_idx]->id()
                              << " (state=" << lm.state << ") dist=" << dist_stream.str()
                              << " | pred " << format_bbox(pred_box) << "\n";
                }
            }
            if (!assoc.unmatched_dets.empty()) {
                std::cout << "    Unmatched detections:\n";
                for (int local_idx : assoc.unmatched_dets) {
                    if (local_idx < 0 || local_idx >= static_cast<int>(det_indexes.size())) continue;
                    int det_global = det_indexes[local_idx];
                    const BBox& det_box = frame.dets[det_global];
                    std::string reason = "no active tracks";
                    if (dist_ptr && local_idx < dist_ptr->rows() && dist_ptr->cols() > 0) {
                        double best_dist = std::numeric_limits<double>::infinity();
                        int best_trk = -1;
                        for (int col = 0; col < dist_ptr->cols(); ++col) {
                            double d = (*dist_ptr)(local_idx, col);
                            if (d < best_dist) {
                                best_dist = d;
                                best_trk = col;
                            }
                        }
                        if (best_trk >= 0 && best_trk < static_cast<int>(trackers_.size())) {
                            if (best_dist > dist_threshold) {
                                reason = "all distances > thresh";
                            } else {
                                std::ostringstream rs;
                                rs << "closest track#" << trackers_[best_trk]->id()
                                   << " dist=" << std::fixed << std::setprecision(3) << best_dist;
                                reason = rs.str();
                            }
                        }
                    }
                    std::cout << "      det#" << det_global << " " << format_bbox(det_box)
                              << " | " << reason << "\n";
                }
            }
            if (!assoc.unmatched_tracks.empty()) {
                std::cout << "    Unmatched tracks:\n";
                for (int trk_idx : assoc.unmatched_tracks) {
                    if (trk_idx < 0 || trk_idx >= static_cast<int>(trackers_.size())) continue;
                    const auto& trk = trackers_[trk_idx];
                    const auto& lm = trk->life_manager();
                    const BBox& pred_box = trk_predictions[trk_idx];
                    std::string reason = "no dets";
                    if (dist_ptr && dist_ptr->rows() > 0 && trk_idx < dist_ptr->cols()) {
                        double best_dist = std::numeric_limits<double>::infinity();
                        int best_det = -1;
                        for (int row = 0; row < dist_ptr->rows(); ++row) {
                            double d = (*dist_ptr)(row, trk_idx);
                            if (d < best_dist) {
                                best_dist = d;
                                best_det = row;
                            }
                        }
                    if (best_det >= 0 && best_det < static_cast<int>(det_indexes.size())) {
                        std::cout << "all distances = " << best_dist
                                  << " , threshold = " << dist_threshold << "\n";
                        if (best_dist > dist_threshold) {
                            reason = "all distances > thresh";
                        } else {
                                std::ostringstream rs;
                                rs << "closest det#" << det_indexes[best_det]
                                   << " dist=" << std::fixed << std::setprecision(3) << best_dist;
                                reason = rs.str();
                            }
                        }
                    }
                    std::cout << "      track#" << trk->id() << " " << format_bbox(pred_box)
                              << " (state=" << lm.state
                              << ", time_since_update=" << lm.time_since_update << ")"
                              << " | " << reason << "\n";
                }
            }
        }
        std::cout << std::defaultfloat;
    }

    std::vector<int> unmatched_tracks = assoc.unmatched_tracks;
    std::vector<int> unmatched_dets;
    unmatched_dets.reserve(assoc.unmatched_dets.size());
    for (int local_idx : assoc.unmatched_dets) {
        if (local_idx >= 0 && local_idx < static_cast<int>(det_indexes.size())) {
            unmatched_dets.push_back(det_indexes[local_idx]);
        }
    }

    auto t_update_start = std::chrono::high_resolution_clock::now();
    update_matched_tracks(frame, assoc.matches, det_indexes, dist_ptr);
    update_unmatched_tracks(frame, unmatched_tracks, trk_predictions, dist_ptr);
    spawn_tracks(frame, unmatched_dets);
    prune_tracks(debug_lifecycle_);
    apply_post_nms(configs_, frame_count_, trackers_, debug_lifecycle_);
    auto t_update_end = std::chrono::high_resolution_clock::now();

        if (debug_lifecycle_) {
            std::cout << "  [Active Tracks]\n";
            for (const auto& trk : trackers_) {
                std::cout << "    track#" << trk->id()
                          << " type=" << trk->det_type()
                          << " " << format_bbox(trk->get_state())
                          << " state=" << trk->state_string(frame_count_) << "\n";
            }
            std::cout << std::defaultfloat;
        }

    last_time_stamp_ = frame.time_stamp;

    std::vector<TrackResult> results;
    for (const auto& trk : trackers_) {
        if (!trk->valid_output(frame_count_)) {
            continue;
        }
        TrackResult r;
        r.bbox = trk->get_state();
        double score = r.bbox.has_score() ? r.bbox.s : std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(score) || score <= 0.0) {
            continue;  // filter out zero/invalid score tracks from final output
        }
        r.id = trk->id();
        r.state = trk->state_string(frame_count_);
        r.det_type = trk->det_type();
        results.push_back(r);
    }

    if (debug_association_ || debug_lifecycle_) {
        std::cout << "-----------------------\n";
    }

    {
        auto predict_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_predict_end - t_predict_start).count();
        auto assoc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_assoc_end - t_assoc_start).count();
        auto dist_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_dist_end - t_dist_start).count();
        auto update_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_update_end - t_update_start).count();
        auto total_ms = predict_ms + assoc_ms + dist_ms + update_ms;
        if (timing) {
            timing->predict_ms = static_cast<double>(predict_ms);
            timing->assoc_ms = static_cast<double>(assoc_ms);
            timing->dist_ms = static_cast<double>(dist_ms);
            timing->update_ms = static_cast<double>(update_ms);
        }
        std::cout << "[TrackTiming] predict=" << predict_ms
                  << "ms assoc=" << assoc_ms
                  << "ms dist=" << dist_ms
                  << "ms update=" << update_ms
                  << "ms total=" << total_ms << "ms\n";
    }
    return results;
}

}  // namespace simpletrack
