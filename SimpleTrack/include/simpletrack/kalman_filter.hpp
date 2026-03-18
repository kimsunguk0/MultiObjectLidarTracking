// Kalman filter motion model translated from mot_3d/motion_model/kalman_filter.py
#pragma once

#include <Eigen/Dense>

#include <optional>
#include <vector>

#include "simpletrack/bbox.hpp"

namespace simpletrack {

struct KalmanFilterParams {
    std::optional<Eigen::Matrix<double, 7, 7>> measurement_noise;
    std::optional<Eigen::Matrix<double, 10, 10>> initial_covariance;
};

class KalmanFilterMotionModel {
public:
    KalmanFilterMotionModel(const BBox& bbox,
                            int inst_type,
                            double time_stamp,
                            const KalmanFilterParams& params = {},
                            bool enable_debug = false);

    void update(const BBox& det_bbox);
    BBox get_prediction(double time_stamp);
    BBox get_state() const;
    void override_yaw(double yaw);
    Eigen::Matrix<double, 7, 7> compute_innovation_matrix() const;
    void sync_time_stamp(double time_stamp);
    int inst_type() const { return inst_type_; }
    void set_debug_id(int id);

private:
    void predict_step();
    void apply_measurement(const Eigen::Matrix<double, 7, 1>& z);
    void normalize_angle(double& angle) const;
    void normalize_state_angle(Eigen::Matrix<double, 10, 1>& x) const;

    double prev_time_stamp_{0.0};
    double latest_time_stamp_{0.0};
    double score_{0.0};
    int inst_type_{0};
    int debug_id_{-1};
    bool debug_enabled_{false};

    Eigen::Matrix<double, 10, 1> x_;
    Eigen::Matrix<double, 10, 10> P_;
    Eigen::Matrix<double, 10, 10> F_;
    Eigen::Matrix<double, 10, 10> Q_;
    Eigen::Matrix<double, 7, 10> H_;
    Eigen::Matrix<double, 7, 7> R_;

    std::vector<BBox> history_;
};

}  // namespace simpletrack
