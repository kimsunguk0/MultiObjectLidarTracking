// C++ scaffold generated for mot_3d/motion_model/kalman_filter.py
#include <stdexcept>
#include <string>

namespace simpletrack {
void placeholder_mot_3d_motion_model_kalman_filter_py() {
    throw std::logic_error("TODO: translate mot_3d/motion_model/kalman_filter.py into C++");
}
}
#include "simpletrack/kalman_filter.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace simpletrack {

namespace {
constexpr int kStateDim = 10;
constexpr int kMeasureDim = 7;
constexpr double kDefaultDt = 0.1;

Eigen::Matrix<double, kStateDim, kStateDim> make_process_noise() {
    Eigen::Matrix<double, kStateDim, kStateDim> Q = Eigen::Matrix<double, kStateDim, kStateDim>::Zero();
    const double q_pos = 0.01;
    Q(0, 0) = Q(1, 1) = Q(2, 2) = q_pos;
    Q(3, 3) = 0.1;
    const double q_vel = 0.1;
    Q(7, 7) = Q(8, 8) = Q(9, 9) = q_vel;
    return Q;
}

Eigen::Matrix<double, kStateDim, kStateDim> make_transition_matrix(double dt) {
    Eigen::Matrix<double, kStateDim, kStateDim> F = Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    F(0, 7) = dt;
    F(1, 8) = dt;
    F(2, 9) = dt;
    return F;
}

Eigen::Matrix<double, kMeasureDim, kStateDim> make_measurement_matrix() {
    Eigen::Matrix<double, kMeasureDim, kStateDim> H = Eigen::Matrix<double, kMeasureDim, kStateDim>::Zero();
    for (int i = 0; i < kMeasureDim; ++i) {
        H(i, i) = 1.0;
    }
    return H;
}

Eigen::Matrix<double, kMeasureDim, kMeasureDim> make_identity_measurement_noise() {
    Eigen::Matrix<double, kMeasureDim, kMeasureDim> R = Eigen::Matrix<double, kMeasureDim, kMeasureDim>::Identity();
    return R;
}

}  // namespace

KalmanFilterMotionModel::KalmanFilterMotionModel(const BBox& bbox,
                                                 int inst_type,
                                                 double time_stamp,
                                                 const KalmanFilterParams& params,
                                                 bool enable_debug)
    : prev_time_stamp_(time_stamp),
      latest_time_stamp_(time_stamp),
      score_(bbox.has_score() ? bbox.s : 0.0),
      inst_type_(inst_type),
      debug_enabled_(enable_debug),
      x_(Eigen::Matrix<double, kStateDim, 1>::Zero()),
      P_(Eigen::Matrix<double, kStateDim, kStateDim>::Identity()),
      F_(make_transition_matrix(kDefaultDt)),
      Q_(make_process_noise()),
      H_(make_measurement_matrix()),
      R_(make_identity_measurement_noise()) {

    auto arr = bbox.to_array();
    for (int i = 0; i < kMeasureDim; ++i) {
        x_(i, 0) = arr[i];
    }
    // velocities initialized to zero
    P_.block<3, 3>(7, 7) *= 1000.0;
    P_ *= 10.0;

    if (params.measurement_noise) {
        R_ = *params.measurement_noise;
    }
    if (params.initial_covariance) {
        P_ = *params.initial_covariance;
    }

    history_.push_back(bbox);
}

void KalmanFilterMotionModel::normalize_angle(double& angle) const {
    while (angle >= M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
}

void KalmanFilterMotionModel::normalize_state_angle(Eigen::Matrix<double, kStateDim, 1>& x) const {
    double angle = x(3);
    normalize_angle(angle);
    x(3) = angle;
}

void KalmanFilterMotionModel::set_debug_id(int id) {
    debug_id_ = id;
    if (debug_enabled_ && debug_id_ >= 0) {
        std::cout << "[KF-config] track#" << debug_id_
                  << " R diag:";
        for (int i = 0; i < kMeasureDim; ++i) {
            std::cout << " " << R_(i, i);
        }
        std::cout << " | yaw_R=" << R_(3, 3) << "\n";
    }
}

void KalmanFilterMotionModel::predict_step() {
    double pre_yaw = x_(3);
    // if (debug_id_ >= 0) {
    //     std::cout << "[KF-predict] track#" << debug_id_
    //               << " pre-o=" << pre_yaw << "\n";
    // }
    x_ = F_ * x_;
    normalize_state_angle(x_);
    double post_yaw = x_(3);
    // if (debug_id_ >= 0) {
    //     std::cout << "[KF-predict] track#" << debug_id_
    //               << " post-o=" << post_yaw << "\n";
    // }
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilterMotionModel::apply_measurement(const Eigen::Matrix<double, kMeasureDim, 1>& z_in) {
    Eigen::Matrix<double, kMeasureDim, 1> z = z_in;

    normalize_state_angle(x_);
    double predicted_theta = x_(3);
    double meas_theta = z(3);
    normalize_angle(meas_theta);
    z(3) = meas_theta;

    double diff = std::fabs(meas_theta - predicted_theta);
    if (diff > M_PI / 2.0 && diff < M_PI * 3.0 / 2.0) {
        x_(3) += (meas_theta > predicted_theta) ? M_PI : -M_PI;
        normalize_state_angle(x_);
    }
    diff = std::fabs(meas_theta - x_(3));
    if (diff >= M_PI * 3.0 / 2.0) {
        x_(3) += (meas_theta > 0.0) ? 2.0 * M_PI : -2.0 * M_PI;
    }

    Eigen::Matrix<double, kMeasureDim, 1> y = z - H_ * x_;
    Eigen::Matrix<double, kMeasureDim, kMeasureDim> S = H_ * P_ * H_.transpose() + R_;
    Eigen::Matrix<double, kStateDim, kMeasureDim> K = P_ * H_.transpose() * S.inverse();


    double pre_yaw = x_(3);
    double meas_yaw = z(3);

    if (debug_enabled_ && debug_id_ >= 0) {
        std::cout << "[KF-update] track#" << debug_id_
                  << " pre-o=" << pre_yaw
                  << " meas-o=" << meas_yaw
                  << " | H(3,3)=" << H_(3, 3)
                  << " P(3,3)=" << P_(3, 3)
                  << " R(3,3)=" << R_(3, 3)
                  << " S(3,3)=" << S(3, 3)
                  << " K(3,3)=" << K(3, 3)
                   << " y(3,3)=" << y(3)
                  << "\n";
    }

    x_ = x_ + K * y;
    normalize_state_angle(x_);
    double post_yaw = x_(3);
    if (debug_enabled_ && debug_id_ >= 0) {
        std::cout << "[KF-update] track#" << debug_id_
                  << " post-o=" << post_yaw << "\n";
    }
    Eigen::Matrix<double, kStateDim, kStateDim> I = Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    P_ = (I - K * H_) * P_;
}

void KalmanFilterMotionModel::update(const BBox& det_bbox) {
    Eigen::Matrix<double, kMeasureDim, 1> z = Eigen::Matrix<double, kMeasureDim, 1>::Zero();
    auto arr = det_bbox.to_array();
    for (int i = 0; i < kMeasureDim; ++i) {
        z(i) = arr[i];
    }

    predict_step();

    apply_measurement(z);

    prev_time_stamp_ = latest_time_stamp_;
    normalize_state_angle(x_);

    if (det_bbox.has_score()) {
        score_ = det_bbox.s;
    } else {
        score_ *= 0.01;
    }

    if (!history_.empty()) {
        auto array = x_.head<7>();
        std::vector<double> data(8);
        for (int i = 0; i < 7; ++i) {
            data[i] = array(i);
        }
        data[7] = score_;
        history_.back() = BBox::from_vector(data);
    }
}

BBox KalmanFilterMotionModel::get_prediction(double time_stamp) {
    latest_time_stamp_ = time_stamp;
    const double dt = kDefaultDt;
    F_ = make_transition_matrix(dt);
    predict_step();

    Eigen::Matrix<double, 7, 1> arr = x_.head<7>();
    std::vector<double> data(8);
    for (int i = 0; i < 7; ++i) {
        data[i] = arr(i);
    }
    data[7] = score_;
    BBox bbox = BBox::from_vector(data);
    bbox.s = score_;
    history_.push_back(bbox);
    return bbox;
}

BBox KalmanFilterMotionModel::get_state() const {
    if (history_.empty()) {
        throw std::runtime_error("History is empty");
    }
    return history_.back();
}

void KalmanFilterMotionModel::override_yaw(double yaw) {
    normalize_angle(yaw);
    x_(3) = yaw;
    normalize_state_angle(x_);
    if (!history_.empty()) {
        history_.back().o = yaw;
    }
}

Eigen::Matrix<double, 7, 7> KalmanFilterMotionModel::compute_innovation_matrix() const {
    return H_ * P_ * H_.transpose() + R_;
}

void KalmanFilterMotionModel::sync_time_stamp(double time_stamp) {
    prev_time_stamp_ = time_stamp;
    latest_time_stamp_ = time_stamp;
}

}  // namespace simpletrack
