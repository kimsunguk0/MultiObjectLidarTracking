// Simplified MOTModel translated from mot_3d/mot.py
#pragma once

#include <Eigen/Dense>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "simpletrack/association.hpp"
#include "simpletrack/config.hpp"
#include "simpletrack/frame_data.hpp"
#include "simpletrack/redundancy.hpp"
#include "simpletrack/tracklet.hpp"

namespace simpletrack {

struct TrackResult {
    BBox bbox;
    int id{0};
    std::string state;
    int det_type{0};
};

struct TrackTiming {
    double predict_ms{0.0};
    double assoc_ms{0.0};
    double dist_ms{0.0};
    double update_ms{0.0};
};

class MOTModel {
public:
    explicit MOTModel(TrackerConfig configs);

    void set_debug(bool association = false, bool lifecycle = false);

    std::vector<TrackResult> frame_mot(const FrameData& frame, TrackTiming* timing = nullptr);

    const std::vector<std::unique_ptr<Tracklet>>& trackers() const { return trackers_; }

private:
    TrackerConfig configs_;
    std::vector<std::unique_ptr<Tracklet>> trackers_;
    RedundancyModule redundancy_;
    int frame_count_{0};
    int next_track_id_{0};
    double last_time_stamp_{0.0};
    bool has_time_stamp_{false};

    bool debug_association_{false};
    bool debug_lifecycle_{false};

    std::vector<BBox> predict_tracks(double time_stamp, bool is_key_frame);
    void update_matched_tracks(const FrameData& frame,
                               const std::vector<std::pair<int, int>>& matches,
                               const std::vector<int>& det_indexes,
                               const Eigen::MatrixXd* dist_matrix);
    void update_unmatched_tracks(const FrameData& frame,
                                 const std::vector<int>& unmatched_tracks,
                                 const std::vector<BBox>& predicted,
                                 const Eigen::MatrixXd* dist_matrix);
    void spawn_tracks(const FrameData& frame,
                      const std::vector<int>& unmatched_dets);
    void prune_tracks(bool log_active_tracks);
};

}  // namespace simpletrack
