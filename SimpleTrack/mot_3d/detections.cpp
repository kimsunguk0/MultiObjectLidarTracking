#include "simpletrack/detections.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "simpletrack/config.hpp"

namespace simpletrack {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

std::pair<double, double> px_to_ego(double v_px, double u_px, const Boundary& boundary) {
    const double d = boundary.discretization();
    double x_e = boundary.minX + (v_px + 0.5) * d;
    double y_img = boundary.minY + (u_px + 0.5) * d;
    double y_e = -y_img;
    return {x_e, y_e};
}

int class_token(int cls_idx) {
    switch (cls_idx) {
        case 0: return 2;  // pedestrian
        case 1: return 1;  // vehicle
        case 2: return 4;  // cyclist
        default: return 1;
    }
}

double normalize_yaw(double yaw) {
    yaw = std::fmod(yaw + kPi, kTwoPi);
    if (yaw < 0) yaw += kTwoPi;
    return yaw - kPi;
}

}  // namespace

FrameData detections_to_framedata(const std::vector<RawDetection>& detections,
                                  double timestamp,
                                  const Eigen::Matrix4d& ego,
                                  bool is_key_frame,
                                  const std::string& cls_name,
                                  const Boundary& boundary) {
    std::vector<BBox> dets;
    dets.reserve(detections.size());
    std::vector<int> det_types;
    det_types.reserve(detections.size());

    const double d = boundary.discretization();
    for (const auto& det : detections) {
        auto [x_e, y_e] = px_to_ego(det.y_px, det.x_px, boundary);
        double w_m = det.w_px * d;
        double l_m = det.l_px * d;
        double yaw = normalize_yaw(det.yaw_rad);
        // std::cout << "[detections_to_framedata] cls=" << det.cls << " yaw_px=" << det.yaw_rad
        //           << " -> o_e=" << yaw << std::endl;

        BBox bbox;
        bbox.x = x_e;
        bbox.y = y_e;
        bbox.z = det.z_m;
        bbox.h = det.h_m;
        bbox.w = w_m;
        bbox.l = l_m;
        bbox.o = yaw;
        bbox.s = det.score;
        dets.push_back(bbox);
        det_types.push_back(class_token(det.cls));
    }

    AuxInfo aux;
    aux.is_key_frame = is_key_frame;
    if (!cls_name.empty()) {
        aux.has_cls_name = true;
        aux.cls_name = cls_name;
    }

    return FrameData(std::move(dets), ego, timestamp, std::move(det_types), aux);
}

}  // namespace simpletrack
