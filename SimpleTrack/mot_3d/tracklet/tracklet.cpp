#include "simpletrack/tracklet.hpp"

#include <Eigen/Dense>

#include <stdexcept>
#include <cmath>
#include <iostream>

namespace simpletrack {

namespace {
KalmanFilterParams build_kf_params(const TrackerConfig& configs) {
    KalmanFilterParams params;
    if (configs.running.measurement_noise) {
        params.measurement_noise = configs.running.measurement_noise;
    }
    if (configs.running.covariance) {
        params.initial_covariance = configs.running.covariance;
    }
    return params;
}
}  // namespace

double Tracklet::unwrap_continuous(double prev, double curr) {
    double delta = curr - prev;
    while (delta > M_PI) {
        curr -= 2.0 * M_PI;
        delta = curr - prev;
    }
    while (delta < -M_PI) {
        curr += 2.0 * M_PI;
        delta = curr - prev;
    }
    return curr;
}

double Tracklet::smoothed_yaw(double default_yaw) const {
    if (yaw_history_.size() < 2) return default_yaw;
    const double t_now = yaw_history_.back().t;
    const double yaw_wrap_to_pi = [](double ang) {
        double a = ang;
        while (a >= M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }(0.0);  // dummy to avoid unused warning

    double best_a = 0.0;
    double best_b = 0.0;
    int best_inliers = 0;
    double best_error = std::numeric_limits<double>::infinity();
    constexpr double thresh = 0.2;  // rad

    for (size_t i = 0; i + 1 < yaw_history_.size(); ++i) {
        for (size_t j = i + 1; j < yaw_history_.size(); ++j) {
            double t1 = yaw_history_[i].t;
            double t2 = yaw_history_[j].t;
            if (t2 == t1) continue;
            double y1 = yaw_history_[i].yaw;
            double y2 = yaw_history_[j].yaw;
            double a = (y2 - y1) / (t2 - t1);
            double b = y1 - a * t1;

            int inliers = 0;
            double error_sum = 0.0;
            for (const auto& s : yaw_history_) {
                double y_pred = a * s.t + b;
                double err = std::fabs(y_pred - s.yaw);
                if (err < thresh) {
                    inliers++;
                    error_sum += err;
                }
            }
            if (inliers > best_inliers ||
                (inliers == best_inliers && error_sum < best_error)) {
                best_inliers = inliers;
                best_error = error_sum;
                best_a = a;
                best_b = b;
            }
        }
    }
    double yaw_est = best_a * t_now + best_b;
    while (yaw_est >= M_PI) yaw_est -= 2.0 * M_PI;
    while (yaw_est < -M_PI) yaw_est += 2.0 * M_PI;
    return yaw_est;
}

Tracklet::Tracklet(const TrackerConfig& configs,
                   int id,
                   const BBox& bbox,
                   int det_type,
                   int frame_index,
                   double time_stamp,
    AuxInfo aux_info,
    bool enable_debug)
    : id_(id),
      time_stamp_(time_stamp),
      asso_(configs.running.asso),
      configs_(configs),
      det_type_(det_type),
      aux_info_(std::move(aux_info)),
      debug_enabled_(enable_debug),
      life_manager_(configs, frame_index),
      latest_score_(bbox.has_score() ? bbox.s : 0.0) {

    if (configs.running.motion_model == "kf") {
        motion_model_ = std::make_unique<KalmanFilterMotionModel>(bbox, det_type_, time_stamp_, build_kf_params(configs_), enable_debug);
        motion_model_->set_debug_id(enable_debug ? id_ : -1);
    } else {
        throw std::invalid_argument("Unsupported motion model: " + configs.running.motion_model);
    }

    double yaw = bbox.o;
    if (yaw_history_.empty()) {
        yaw_history_.push_back({static_cast<double>(frame_index), yaw});
    }
}

BBox Tracklet::predict(double time_stamp, bool is_key_frame) {
    if (!motion_model_) {
        throw std::runtime_error("Motion model not initialized");
    }
    BBox result = motion_model_->get_prediction(time_stamp);
    life_manager_.predict(is_key_frame);
    latest_score_ *= 0.01;
    result.s = latest_score_;
    return result;
}

void Tracklet::update(const UpdateInfoData& update_info) {
    if (!motion_model_) {
        throw std::runtime_error("Motion model not initialized");
    }
    latest_score_ = update_info.bbox.s;
    bool is_key_frame = update_info.aux_info.is_key_frame;

    if (update_info.mode == 1 || update_info.mode == 3) {
        motion_model_->update(update_info.bbox);
    }
    life_manager_.update(update_info, is_key_frame);

    double yaw_new = motion_model_->get_state().o;
    if (!yaw_history_.empty()) {
        yaw_new = unwrap_continuous(yaw_history_.back().yaw, yaw_new);
    }
    yaw_history_.push_back({static_cast<double>(update_info.frame_index), yaw_new});
    if (yaw_history_.size() > kMaxYawHistory) {
        yaw_history_.pop_front();
    }
}

BBox Tracklet::get_state() const {
    BBox result = motion_model_->get_state();
    result.s = latest_score_;
    return result;
}

bool Tracklet::valid_output(int frame_index) const {
    return life_manager_.valid_output(frame_index);
}

bool Tracklet::death(int frame_index) const {
    return life_manager_.death(frame_index);
}

std::string Tracklet::state_string(int frame_index) const {
    return life_manager_.state_string(frame_index);
}

Eigen::Matrix<double, 7, 7> Tracklet::compute_innovation_matrix() const {
    return motion_model_->compute_innovation_matrix();
}

void Tracklet::sync_time_stamp(double time_stamp) {
    time_stamp_ = time_stamp;
    if (motion_model_) {
        motion_model_->sync_time_stamp(time_stamp);
    }
}

}  // namespace simpletrack
