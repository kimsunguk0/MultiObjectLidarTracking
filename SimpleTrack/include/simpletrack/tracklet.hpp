// Tracklet management translated from mot_3d/tracklet/tracklet.py
#pragma once

#include <Eigen/Dense>

#include <memory>
#include <deque>
#include <limits>

#include "simpletrack/config.hpp"
#include "simpletrack/hit_manager.hpp"
#include "simpletrack/kalman_filter.hpp"
#include "simpletrack/update_info_data.hpp"

namespace simpletrack {

class Tracklet {
public:
    Tracklet(const TrackerConfig& configs,
             int id,
             const BBox& bbox,
             int det_type,
             int frame_index,
             double time_stamp = 0.0,
             AuxInfo aux_info = AuxInfo(),
             bool enable_debug = false);

    BBox predict(double time_stamp = 0.0, bool is_key_frame = true);
    void update(const UpdateInfoData& update_info);
    BBox get_state() const;
    bool valid_output(int frame_index) const;
    bool death(int frame_index) const;
    std::string state_string(int frame_index) const;
    Eigen::Matrix<double, 7, 7> compute_innovation_matrix() const;
    void sync_time_stamp(double time_stamp);

    int id() const { return id_; }
    int det_type() const { return det_type_; }
    double latest_score() const { return latest_score_; }
    const HitManager& life_manager() const { return life_manager_; }
    HitManager& mutable_life_manager() { return life_manager_; }
    const KalmanFilterMotionModel& motion_model() const { return *motion_model_; }
    KalmanFilterMotionModel& mutable_motion_model() { return *motion_model_; }

private:
    int id_{0};
    double time_stamp_{0.0};
    std::string asso_;
    TrackerConfig configs_;
    int det_type_{0};
    AuxInfo aux_info_;
    std::unique_ptr<KalmanFilterMotionModel> motion_model_;
    HitManager life_manager_;
    double latest_score_{0.0};
    bool debug_enabled_{false};

    struct YawSample {
        double t{0.0};
        double yaw{0.0};
    };
    std::deque<YawSample> yaw_history_;
    static constexpr size_t kMaxYawHistory = 10;

    static double unwrap_continuous(double prev, double curr);
    double smoothed_yaw(double default_yaw) const;
};

}  // namespace simpletrack
